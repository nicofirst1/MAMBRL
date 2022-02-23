import math
from typing import Dict

import torch
from tqdm import trange

from agent.PPO_Agent import PPO_Agent
from agent.RolloutStorage import RolloutStorage
from common import mas_dict2tensor
from logging_callbacks.wandbLogger import preprocess_logs
from src.agent.PpoWrapper import PpoWrapper
from src.common import Params
from src.env.EnvWrapper import get_env_wrapper
from src.model.ModelFree import ModelFree
from src.trainer.BaseTrainer import BaseTrainer


class ModelFreeTrainer(BaseTrainer):
    def __init__(self, model, agent, env, config: Params):
        """__init__ method


        Parameters
        ----------
        model : ModelFree class
            one of the ModelFree classes defined in the src.model directory
        agent : agent class
            one of the agent classes defined in the src.agent directory
        env : env class
            one of the env classes defined in the src.env directory
        config : Params
            instance of the class Params defined in src.common.Params

        Returns
        -------
        None.

        """
        super(ModelFreeTrainer, self).__init__(env, config)

        self.num_agents = config.agents
        self.num_episodes = config.episodes
        self.num_steps = config.num_steps

        self.action_space = config.num_actions
        self.frame_shape = config.frame_shape
        self.obs_shape = config.obs_shape

        # Build a ppo agent for each agent in the env
        ppo_configs = config.get_ppo_configs()
        model_free_configs = config.get_model_free_configs()
        self.ppo_agents = {
            agent_id: agent(model, config.device, model_free_configs, **ppo_configs)
            for agent_id in self.cur_env.agents
        }

        self.use_wandb = config.use_wandb
        if self.use_wandb:
            from logging_callbacks import PPOWandb
            cams = []

            self.logger = PPOWandb(
                train_log_step=5,
                val_log_step=5,
                project="model_free",
                opts={},
                models=self.ppo_agents["agent_0"].get_modules(),
                horizon=config.horizon,
                action_meaning=self.cur_env.env.action_meaning_dict,
                cams=cams,
            )

    def set_env(self, new_env):
        self.cur_env = new_env

    def collect_trajectories(self) -> RolloutStorage:
        [model.eval() for model in self.ppo_agents.values()]

        rollout = RolloutStorage(
            num_steps=self.num_steps * self.num_episodes,
            frame_shape=self.frame_shape,
            obs_shape=self.obs_shape,
            num_actions=self.action_space,
            num_agents=self.num_agents,
        )
        rollout.to(self.config.device)

        action_dict = {agent_id: False for agent_id in self.cur_env.agents}
        values_dict = {agent_id: False for agent_id in self.cur_env.agents}
        action_log_dict = {agent_id: False for agent_id in self.cur_env.agents}

        for episode in range(self.num_episodes):
            observation = self.cur_env.reset()
            rollout.states[episode * self.num_steps] = observation.unsqueeze(dim=0)

            for step in range(self.num_steps):
                obs = observation.to(self.config.device).unsqueeze(dim=0)

                for agent_id in self.cur_env.agents:
                    # FIX: the mask of the rollout is a different thing from the mask of the rnn
                    with torch.no_grad():
                        value, action, action_log_prob = self.ppo_agents[agent_id].act(
                            obs, rollout.masks[step]
                        )

                    action_dict[agent_id] = action
                    values_dict[agent_id] = value
                    action_log_dict[agent_id] = action_log_prob

                observation, rewards, done, infos = self.cur_env.step(action_dict)

                actions = mas_dict2tensor(action_dict, int)
                action_log_probs = torch.cat([elem.unsqueeze(0) for _, elem in action_log_dict.items()], 0)
                masks = (~torch.tensor(done["__all__"])).float().unsqueeze(0)
                rewards = mas_dict2tensor(rewards, float)
                values = mas_dict2tensor(values_dict, float)

                rollout.insert(
                    state=observation,
                    next_state=None,
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

        num_updates = self.config.ppo_epochs * int(math.ceil(rollout.rewards.size(0) / self.config.minibatch))

        action_losses = sum(action_losses.values()) / num_updates
        value_losses = sum(value_losses.values()) / num_updates
        entropies = sum(entropies.values()) / num_updates

        return action_losses, value_losses, entropies, logs

    def checkpoint(self):
        pass
        # torch.save({
        #     'epoch': EPOCH,
        #     'model_state_dict': net.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': LOSS,
        # }, PATH)

    def restore_training(self):
        pass

if __name__ == '__main__':
    params = Params()
    env = get_env_wrapper(params)

    trainer = ModelFreeTrainer(ModelFree, PPO_Agent, env, params)
    for epoch in trange(params.model_free_epochs, desc="Training model free"):
        rollout = trainer.collect_trajectories()
        action_loss, value_loss, entropy, logs = trainer.train(rollout)

        if params.use_wandb:
            logs = preprocess_logs([value_loss, action_loss, entropy, logs], trainer)
            trainer.logger.on_batch_end(logs=logs, batch_id=epoch, rollout=rollout)

        rollout.after_update()

