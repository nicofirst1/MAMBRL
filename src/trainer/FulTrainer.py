from functools import partial

import torch
from tqdm import trange
from typing import Dict
import math

from src.agent.PPO_Agent import PPO_Agent
from src.agent.RolloutStorage import RolloutStorage
from src.common import Params, mas_dict2tensor
from src.common.utils import one_hot_encode
from src.env.EnvWrapper import get_env_wrapper
from src.model import NextFramePredictor
from src.model.ModelFree import ModelFree, FeatureExtractor
from src.model.FullModel import FullModel
from src.model.RolloutEncoder import RolloutEncoder
from src.trainer.BaseTrainer import BaseTrainer
from src.trainer.EnvModelTrainer import EnvModelTrainer
from src.trainer.ModelFreeTrainer import ModelFreeTrainer
from src.trainer.Policies import MultimodalMAS


class FullTrainer(BaseTrainer):
    def __init__(self, agent, config: Params):
        """__init__ module.

        config : Params
            instance of the class Params defined in src.common.Params

        Returns
        -------
        None.

        """

        env = get_env_wrapper(config)

        super(FullTrainer, self).__init__(env, config)

        self.model = {agent_id: FullModel(FeatureExtractor, ModelFree,
                                          NextFramePredictor, RolloutEncoder, config)
                      for agent_id in self.cur_env.agents}
        self.ppo_agents = {agent_id: agent(model, **config)
                           for agent_id in self.cur_env.agents
                           }
        self.policy = MultimodalMAS(self.model)

    def collect_trajectories(self) -> RolloutStorage:

        rollout = RolloutStorage(
            num_steps=self.config.horizon * self.config.episodes,
            frame_shape=self.config.frame_shape,
            obs_shape=self.config.obs_shape,
            num_actions=self.config.num_actions,
            num_agents=self.config.agents,
        )
        rollout.to(self.config.device)

        if self.logger is not None:
            self.logger.epoch += 1

        action_dict = {agent_id: None for agent_id in self.cur_env.agents}
        done = {agent_id: None for agent_id in self.cur_env.agents}

        for episode in trange(self.config.episodes, desc="Collecting trajectories.."):
            done["__all__"] = False
            observation = self.cur_env.reset()
            rollout.states[episode *
                           self.config.horizon] = observation.unsqueeze(dim=0)

            for step in range(self.config.horizon):
                observation = observation.unsqueeze(
                    dim=0).to(self.config.device)

                for agent_id in self.cur_env.agents:
                    with torch.no_grad():
                        action, _, _ = self.policy.act(agent_id, observation)
                        action_dict[agent_id] = action

                observation, rewards, done, _ = self.cur_env.step(action_dict)

                actions = mas_dict2tensor(action_dict, int)
                rewards = mas_dict2tensor(rewards, float)
                masks = (~torch.tensor(done["__all__"])).float().unsqueeze(0)

                rollout.insert(
                    state=observation,
                    next_state=observation[-3:, :, :],
                    action=actions,
                    action_log_probs=None,
                    value_preds=None,
                    reward=rewards,
                    mask=masks
                )

                if done["__all__"]:
                    rollout.compute_value_world_model(
                        episode * self.config.horizon + step, self.config.gamma)
                    observation = self.cur_env.reset()
        return rollout

    def train(self, rollout: RolloutStorage) -> [torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Dict]]:
        [model.train() for model in self.ppo_agents.values()]

        logs = {ag: None for ag in self.ppo_agents.keys()}

        action_losses = {ag: 0 for ag in self.ppo_agents.keys()}
        value_losses = {ag: 0 for ag in self.ppo_agents.keys()}
        entropies = {ag: 0 for ag in self.ppo_agents.keys()}

        with torch.no_grad():
            next_value = self.ppo_agents["agent_0"].get_value(
                rollout.states[-1].unsqueeze(dim=0), rollout.masks[-1]
            ).detach()

        rollout.compute_returns(next_value, True, self.config.gamma, 0.95)
        advantages = rollout.returns - rollout.value_preds
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        for epoch in range(self.config.ppo_epochs):
            data_generator = rollout.recurrent_generator(
                advantages, minibatch_frames=self.config.minibatch
            )

            for sample in data_generator:
                states_batch, actions_batch, logs_probs_batch, \
                    values_batch, return_batch, masks_batch, adv_targ = sample

                for agent_id in self.ppo_agents.keys():
                    agent_index = int(agent_id[-1])

                    agent_actions = actions_batch[:, agent_index]
                    agent_adv_targ = adv_targ[:, agent_index]
                    agent_log_probs = logs_probs_batch[:, agent_index, :]
                    agent_returns = return_batch[:, agent_index]
                    agent_values = values_batch[:, agent_index]

                    with torch.enable_grad():
                        action_loss, value_loss, entropy, log = self.ppo_agents[agent_id].ppo_step(
                            states_batch, agent_actions, agent_log_probs, agent_values,
                            agent_returns, agent_adv_targ, masks_batch
                        )

                    logs[agent_id] = log

                    action_losses[agent_id] += float(action_loss)
                    value_losses[agent_id] += float(value_loss)
                    entropies[agent_id] += float(entropy)

        num_updates = self.config.ppo_epochs * \
            int(math.ceil(rollout.rewards.size(0) / self.config.minibatch))

        action_losses = sum(action_losses.values()) / num_updates
        value_losses = sum(value_losses.values()) / num_updates
        entropies = sum(entropies.values()) / num_updates

        return action_losses, value_losses, entropies, logs
        self.em_trainer.train(rollout)
        self.collect_features()

    def collect_features(self) -> RolloutStorage:

        for episodes in range(self.config.episodes):
            frames = self.cur_env.reset()

            for step in range(self.config.horizon):

                actions = {}
                features = {}

                frames = frames.to(self.device)
                frames = frames.unsqueeze(dim=0)
                mask = torch.ones(1).to(self.device)

                # for every agent
                for agent in self.cur_env.agents.keys():
                    # get the partial agent policy for acting
                    policy = partial(self.policy.act, agent)
                    # extract features from distillated model free agent
                    mf_feat = self.mf_trainer.ppo_agents[agent].actor_critic.feature_extraction(
                        frames, mask)
                    # get action and encode it
                    action, _, _ = policy(frames)
                    actions[agent] = action
                    action = one_hot_encode(action, self.config.num_actions)
                    action = action.to(self.device).unsqueeze(dim=0)
                    # predict up to N rollout in the future
                    pred_obs, pred_rews = self.em_trainer.env_model[agent].rollout_steps(frames,
                                                                                         policy
                                                                                         )
                    # encode prediction and cat them to other ones
                    em_feat = self.encoder(pred_obs, pred_rews)
                    features[agent] = torch.cat((mf_feat, em_feat), dim=-1)

                    # todo: Francesco add action/values arch

                frames, rewards, done, info = self.cur_env.step(actions)


if __name__ == '__main__':
    params = Params()

    trainer = FullTrainer(params)
    # trainer.collect_features()
    rollout = trainer.collect_trajectories()
    trainer.train(rollout)
