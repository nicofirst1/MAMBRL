import torch
from tqdm import trange
from typing import Dict
import math

from logging_callbacks.wandbLogger import preprocess_logs
from src.agent.PPO_Agent import PPO_Agent
from src.agent.RolloutStorage import RolloutStorage
from src.common import Params, mas_dict2tensor
from src.env.EnvWrapper import get_env_wrapper
from src.model import NextFramePredictor
from src.model.ModelFree import ModelFree, FeatureExtractor
from src.model.FullModel import FullModel
from src.model.RolloutEncoder import RolloutEncoder
from src.trainer.BaseTrainer import BaseTrainer


class FullTrainer(BaseTrainer):
    def __init__(self, agent, env, config: Params):
        """__init__ module.

        config : Params
            instance of the class Params defined in src.common.Params

        Returns
        -------
        None.

        """
        super(FullTrainer, self).__init__(env, config)

        self.model = {
            agent_id: FullModel(env, FeatureExtractor, NextFramePredictor, RolloutEncoder, config)
            for agent_id in self.cur_env.agents
        }

        ppo_configs = config.get_ppo_configs()
        self.ppo_agents = {
            agent_id: agent(self.model[agent_id], config.device, **ppo_configs)
            for agent_id in self.cur_env.agents
        }

        self.use_wandb = self.config.use_wandb
        if self.config.use_wandb:
            import os
            from logging_callbacks import FullWandb
            cams = []

            self.logger = FullWandb(
                train_log_step=5,
                val_log_step=5,
                project="full_training",
                opts={},
                horizon=config.horizon,
                action_meaning=self.cur_env.env.action_meaning_dict,
                cams=cams,
            )

            artifact = self.logger.run.use_artifact('mambrl/env_model/agent_0_EnvModel:v14', type='model')
            artifact_dir = artifact.download()

            self.model["agent_0"].env_model.load_model(os.path.join(artifact_dir, 'env_model.pt'))

    def collect_trajectories(self) -> RolloutStorage:
        [model.eval() for model in self.ppo_agents.values()]

        rollout = RolloutStorage(
            num_steps=self.config.horizon * self.config.episodes,
            frame_shape=self.config.frame_shape,
            obs_shape=self.config.obs_shape,
            num_actions=self.config.num_actions,
            num_agents=self.config.agents,
        )
        rollout.to(self.config.device)

        action_dict = {agent_id: None for agent_id in self.cur_env.agents}
        values_dict = {agent_id: False for agent_id in self.cur_env.agents}
        action_log_dict = {agent_id: False for agent_id in self.cur_env.agents}

        for episode in range(self.config.episodes):
            observation = self.cur_env.reset()
            rollout.states[episode * self.config.horizon] = observation.unsqueeze(dim=0)

            for step in range(self.config.horizon):
                observation = observation.unsqueeze(dim=0).to(self.config.device)

                for agent_id in self.cur_env.agents:
                    with torch.no_grad():
                        value, action, action_log_prob = self.ppo_agents[agent_id].act(
                            observation, rollout.masks[step]
                        )

                    action_dict[agent_id] = action
                    values_dict[agent_id] = value
                    action_log_dict[agent_id] = action_log_prob

                observation, rewards, done, _ = self.cur_env.step(action_dict)

                actions = mas_dict2tensor(action_dict, int)
                action_log_probs = torch.cat([elem.unsqueeze(0) for _, elem in action_log_dict.items()], 0)
                rewards = mas_dict2tensor(rewards, float)
                values = mas_dict2tensor(values_dict, float)
                masks = (~torch.tensor(done["__all__"])).float().unsqueeze(0)

                rollout.insert(
                    state=observation,
                    next_state=observation[-3:, :, :],
                    action=actions,
                    action_log_probs=action_log_probs,
                    value_preds=values,
                    reward=rewards,
                    mask=masks
                )

                if done["__all__"]:
                    observation = self.cur_env.reset()

        return rollout

    def train(self, rollout: RolloutStorage) -> [torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Dict]]:
        [model.train() for model in self.ppo_agents.values()]

        logs = {ag: [] for ag in self.ppo_agents.keys()}

        action_losses = {ag: 0 for ag in self.ppo_agents.keys()}
        value_losses = {ag: 0 for ag in self.ppo_agents.keys()}
        entropies = {ag: 0 for ag in self.ppo_agents.keys()}

        with torch.no_grad():
            next_value = self.ppo_agents["agent_0"].get_value(
                rollout.states[-1].unsqueeze(dim=0), rollout.masks[-1]
            ).detach()

        rollout.compute_returns(next_value, True, self.config.gamma, 0.95)
        advantages = rollout.returns - rollout.value_preds

        for _ in range(self.config.ppo_epochs):
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

                    logs[agent_id].append(log)

                    action_losses[agent_id] += float(action_loss)
                    value_losses[agent_id] += float(value_loss)
                    entropies[agent_id] += float(entropy)

        num_updates = self.config.ppo_epochs * int(math.ceil(rollout.step / self.config.minibatch))

        action_losses = sum(action_losses.values()) / num_updates
        value_losses = sum(value_losses.values()) / num_updates
        entropies = sum(entropies.values()) / num_updates

        return action_losses, value_losses, entropies, logs


if __name__ == '__main__':
    params = Params()
    env = get_env_wrapper(params)

    trainer = FullTrainer(PPO_Agent, env, params)

    for epoch in trange(params.model_free_epochs, desc="Training full free"):
        rollout = trainer.collect_trajectories()
        action_losses, value_losses, entropies, logs = trainer.train(rollout)

        if trainer.use_wandb and epoch % params.log_step == 0:
            logs = preprocess_logs([value_losses, action_losses, entropies, logs], trainer)
            trainer.logger.on_batch_end(logs=logs, batch_id=epoch, rollout=rollout)
