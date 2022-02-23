from functools import partial

import torch
from tqdm import trange

from src.agent.PpoWrapper import PpoWrapper
from src.agent.RolloutStorage import RolloutStorage
from src.common import Params, mas_dict2tensor
from src.common.utils import one_hot_encode
from src.env.EnvWrapper import get_env_wrapper
from src.model import NextFramePredictor
from src.model.ModelFree import ModelFree
from src.model.RolloutEncoder import RolloutEncoder
from src.trainer.BaseTrainer import BaseTrainer
from src.trainer.EnvModelTrainer import EnvModelTrainer
from src.trainer.ModelFreeTrainer import ModelFreeTrainer
from src.trainer.Policies import MultimodalMAS


class FullTrainer(BaseTrainer):
    def __init__(self, config: Params):
        """__init__ module.

        Parameters
        ----------
        model : NextFramePredictor
            model in src.model.EnvModel 
        env : env class
            one of the env classes defined in the src.env directory
        config : Params
            instance of the class Params defined in src.common.Params

        Returns
        -------
        None.

        """

        env = get_env_wrapper(config)

        super(FullTrainer, self).__init__(env, config)

        self.em_trainer = EnvModelTrainer(NextFramePredictor, self.cur_env, config)

        self.mf_trainer = ModelFreeTrainer(ModelFree, PpoWrapper, env, params)

        self.policy = MultimodalMAS(self.mf_trainer.agent.actor_critic_dict)

        rollout_params = config.get_rollout_encoder_configs()
        self.encoder = RolloutEncoder(**rollout_params).to(self.device)

    def collect_trajectories(self) -> RolloutStorage:

        rollout = RolloutStorage(
            num_steps=self.config.horizon * self.config.episodes,
            frame_shape=self.config.frame_shape,
            obs_shape=self.config.obs_shape,
            num_actions=self.config.num_actions,
            num_agents=1,
        )
        rollout.to(self.config.device)

        if self.logger is not None:
            self.logger.epoch += 1

        action_dict = {agent_id: None for agent_id in self.cur_env.agents}
        done = {agent_id: None for agent_id in self.cur_env.agents}

        for episode in trange(self.config.episodes, desc="Collecting trajectories.."):
            done["__all__"] = False
            observation = self.cur_env.reset()
            rollout.states[episode * self.config.horizon] = observation.unsqueeze(dim=0)

            for step in range(self.config.horizon):
                observation = observation.unsqueeze(dim=0).to(self.config.device)

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
                    rollout.compute_value_world_model(episode * self.config.horizon + step, self.config.gamma)
                    observation = self.cur_env.reset()
        return rollout

    def train(self):
        for epoch in trange(self.config.epochs, desc="Epoch"):
            rollout = self.collect_trajectories()
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
                    mf_feat = self.mf_trainer.agent.actor_critic_dict[agent].feature_extraction(
                        frames, mask)
                    # get action and encode it
                    action, _, _ = policy(frames)
                    actions[agent] = action
                    action = one_hot_encode(action, self.config.num_actions)
                    action = action.to(self.device).unsqueeze(dim=0)

                    # predict up to N rollout in the future
                    pred_obs, pred_rews = self.em_trainer.env_model[agent].rollout_steps(frames,
                                                                                         action,
                                                                                         policy
                                                                                         )
                    # encode prediction and cat them to other ones
                    em_feat = self.encoder(pred_obs, pred_rews)
                    features[agent] = torch.cat((mf_feat, em_feat), dim=-1)

                frames, rewards, done, info = self.cur_env.step(actions)


if __name__ == '__main__':
    params = Params()

    trainer = FullTrainer(params)
    trainer.train()
