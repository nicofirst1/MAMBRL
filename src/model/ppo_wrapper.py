import math
import random

import torch

from src.common import mas_dict2tensor
from .model_free import Policy, ResNetBase, CNNBase
from .ppo import PPO
from .rollout_storage import RolloutStorage


class PpoWrapper:
    def __init__(self, env, config):

        self.env = env
        self.obs_shape = env.obs_shape
        self.action_space = env.action_space
        self.num_agents = config.agents

        self.gamma = config.gamma
        self.device = config.device

        self.num_steps = config.horizon
        self.num_minibatch = config.minibatch

        self.guided_learning_prob = config.guided_learning_prob

        if config.base == "resnet":
            base = ResNetBase
        elif config.base == "cnn":
            base = CNNBase
        else:
            base = None

        self.actor_critic_dict = {
            agent_id: Policy(
                self.obs_shape,
                self.action_space,
                base=base,
                base_kwargs={'recurrent': config.recurrent}
            ).to(self.device) for agent_id in self.env.agents
        }

        self.ppo_agent = PPO(
            actor_critic_dict=self.actor_critic_dict,
            clip_param=config.ppo_clip_param,
            num_minibatch=self.num_minibatch,
            value_loss_coef=config.value_loss_coef,
            entropy_coef=config.entropy_coef,
            lr=config.lr,
            eps=config.eps,
            max_grad_norm=config.max_grad_norm,
            use_clipped_value_loss=config.clip_value_loss
        )

    def get_learning_rate(self):
        lrs = []
        for k, optim in self.ppo_agent.optimizers.items():
            param_group = optim.param_groups[0]
            lrs.append(param_group['lr'])

        return sum(lrs) / len(lrs)

    def set_guided_learning_prob(self, value):
        self.guided_learning_prob = value

    def set_entropy_coeff(self, value):
        self.ppo_agent.entropy_coef = value

    def learn(self, episodes):
        [model.train() for model in self.actor_critic_dict.values()]

        rollout = RolloutStorage(
            num_steps=self.num_steps,
            obs_shape=self.obs_shape,
            num_agents=self.num_agents,
            recurrent_hs_size=self.actor_critic_dict["agent_0"].recurrent_hidden_state_size
        )
        rollout.to(self.device)

        logs = {ag: dict(
            ratio=[],
            surr1=[],
            surr2=[],
            returns=[],
            adv_targ=[],
            perc_surr1=[],
            perc_surr2=[],
            curr_log_porbs=[],
            old_log_probs=[]
        ) for ag in self.actor_critic_dict.keys()}

        # init dicts and reset env
        action_dict = {agent_id: False for agent_id in self.env.agents}
        values_dict = {agent_id: False for agent_id in self.env.agents}
        action_log_dict = {agent_id: False for agent_id in self.env.agents}
        recurrent_hs_dict = {agent_id: False for agent_id in self.env.agents}

        observation = self.env.reset()
        rollout.states[0] = observation.unsqueeze(dim=0)

        for step in range(self.num_steps):
            obs = observation.to(self.device).unsqueeze(dim=0)
            guided_learning = {agent_id: False for agent_id in self.env.agents}

            for agent_id in self.env.agents:
                agent_index = int(agent_id[-1])

                # perform guided learning with scheduler
                if self.guided_learning_prob > random.uniform(0, 1):
                    action, action_log_prob = self.env.optimal_action(agent_id)
                    guided_learning[agent_id] = True
                    value = -1
                else:
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hs = self.actor_critic_dict[agent_id].act(
                            obs, rollout.recurrent_hs[step, agent_index], rollout.masks[step]
                        )

                # get action with softmax and multimodal (stochastic)
                action_dict[agent_id] = int(action)
                values_dict[agent_id] = float(value)
                action_log_dict[agent_id] = float(action_log_prob)
                recurrent_hs_dict[agent_id] = recurrent_hs[0]

            # Obser reward and next obs
            ## fixme: questo con multi agent non funziona, bisogna capire come impostarlo
            new_observation, rewards, done, infos = self.env.step(action_dict)

            # if guided then use actual reward as predicted value
            for agent_id, b in guided_learning.items():
                if b:
                    values_dict[agent_id] = rewards[agent_id]

            masks = (~torch.tensor(done["__all__"])).float().unsqueeze(0)
            rewards = mas_dict2tensor(rewards, float)
            actions = mas_dict2tensor(action_dict, int)
            values = mas_dict2tensor(values_dict, float)
            recurrent_hs = mas_dict2tensor(recurrent_hs_dict, list)
            action_log_probs = mas_dict2tensor(action_log_dict, float)

            # update observation
            observation = new_observation

            rollout.insert(
                state=observation,
                recurrent_hs=recurrent_hs,
                action=actions,
                action_log_probs=action_log_probs,
                value_preds=values,
                reward=rewards,
                mask=masks
            )

            if done["__all__"]:
                observation = self.env.reset()

        ## fixme: qui bisogna come farlo per multi agent
        with torch.no_grad():
            next_value = self.actor_critic_dict["agent_0"].get_value(
                rollout.states[-1], rollout.recurrent_hs[-1, 0], rollout.masks[-1]).detach()

        rollout.compute_returns(next_value, True, self.gamma, 0.95)

        #with torch.enable_grad():
        value_loss, action_loss, entropy = self.ppo_agent.update(rollout, logs)
        rollout.after_update()

        return value_loss, action_loss, entropy, rollout, logs

    def set_env(self, env):
        self.env = env
