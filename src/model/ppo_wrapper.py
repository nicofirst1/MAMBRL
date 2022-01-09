import torch

from .ppo import PPO
from .rollout_storage import RolloutStorage
from .model_free import Policy, ResNetBase, CNNBase
from src.common import mas_dict2tensor


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

        if config.base == "resnet":
            base = ResNetBase
        elif config.base == "cnn":
            base = CNNBase
        else:
            base = None

        self.actor_critic_dict = {
            agent_id: Policy(self.obs_shape, self.action_space, base=base).to(self.device) for agent_id in self.env.agents
        }

        self.agent = PPO(
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

    def learn(self, episodes, full_log_prob=False, entropy_coef=None):

        rollout = RolloutStorage(
            num_steps=self.num_steps,
            obs_shape=self.obs_shape,
            num_actions=self.action_space,
            num_agents=self.num_agents
        )
        rollout.to(self.device)

        if entropy_coef is not None:
            self.agent.entropy_coef = entropy_coef

        for episode in range(episodes):
            # init dicts and reset env
            action_dict = {agent_id: False for agent_id in self.env.agents}
            values_dict = {agent_id: False for agent_id in self.env.agents}
            action_log_dict = {agent_id: False for agent_id in self.env.agents}

            observation = self.env.reset()
            rollout.states[0] = observation.unsqueeze(dim=0)

            for step in range(self.num_steps):
                normalize_obs = (observation / 255.0).to(self.device).unsqueeze(dim=0)
                for agent_id in self.env.agents:
                    with torch.no_grad():
                        value, action, action_log_prob = self.actor_critic_dict[agent_id].act(
                            normalize_obs, full_log_prob=full_log_prob
                        )

                    # get action with softmax and multimodal (stochastic)
                    action_dict[agent_id] = int(action)
                    values_dict[agent_id] = float(value)
                    if not full_log_prob:
                        action_log_dict[agent_id] = float(action_log_prob)
                    else:
                        action_log_dict[agent_id] = action_log_prob[0]

                # Obser reward and next obs
                ## fixme: questo con multi agent non funziona, bisogna capire come impostarlo
                new_observation, rewards, done, infos = self.env.step(action_dict)

                masks = (~torch.tensor(done["__all__"])).float().unsqueeze(0)

                #masks = 1 - mas_dict2tensor(done, int)
                rewards = mas_dict2tensor(rewards, float)
                actions = mas_dict2tensor(action_dict, int)
                values = mas_dict2tensor(values_dict, float)
                action_log_probs = mas_dict2tensor(
                    action_log_dict, float if not full_log_prob else list
                )

                rollout.insert(
                    step=step,
                    state=observation,
                    action=actions,
                    values=values,
                    reward=rewards,
                    mask=masks,
                    action_log_probs=action_log_probs,
                )

                # update observation
                observation = new_observation

                if done["__all__"]:
                    break

            ## fixme: qui bisogna capire il get_value a cosa serve e come farlo per multi agent
            with torch.no_grad():
                next_value = (
                    self.actor_critic_dict["agent_0"]
                        .get_value(rollout.states[-1].unsqueeze(0))
                        .detach()
                )

            rollout.compute_returns(next_value, True, self.gamma, 0.95)
            value_loss, action_loss, entropy = self.agent.update(rollout)
            rollout.steps = 0

        return value_loss, action_loss, entropy, rollout

    def set_env(self, env):
        self.env = env

    def act(self, obs, agent_id, full_log_prob=False):
        return self.actor_critic_dict[agent_id].act(
            obs, deterministic=True, full_log_prob=True
        )
