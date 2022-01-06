import torch

from model import RolloutStorage, ppo
from model.model_free import Policy
from model.utils import mas_dict2tensor


class PPO:
    def __init__(self, env, obs_shape, action_space, num_agents, device, num_steps=128, gamma=0.99, lr=2.5e-4,
            clip_param=0.1, value_loss_coef=0.5, num_minibatch=4, entropy_coef=0.01, eps=1e-5, max_grad_norm=0.5, ppo_epoch=4):

        self.env = env
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.num_agents = num_agents
        self.device = device

        self.lr = lr
        self.gamma = gamma

        self.num_steps = num_steps
        self.num_minibatch = num_minibatch

        self.actor_critic_dict = {
            agent_id: Policy(obs_shape, action_space).to(device) for agent_id in self.env.agents
        }

        self.agent = ppo.PPO(
            actor_critic_dict=self.actor_critic_dict,
            clip_param=clip_param,
            ppo_epoch=ppo_epoch,
            num_minibatch=num_minibatch,
            value_loss_coef=value_loss_coef,
            entropy_coef=entropy_coef,
            lr=lr,
            eps=eps,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=False
        )

    def learn(self, episodes, full_log_prob=False):

        rollout = RolloutStorage(
            num_steps=self.num_steps,
            obs_shape=self.obs_shape,
            num_actions=self.action_space,
            num_agents=self.num_agents
        )
        rollout.to(self.device)

        for episode in range(episodes):
            # init dicts and reset env
            action_dict = {agent_id: False for agent_id in self.env.agents}
            values_dict = {agent_id: False for agent_id in self.env.agents}
            action_log_dict = {agent_id: False for agent_id in self.env.agents}

            observation = self.env.reset()
            rollout.states[0] = observation.unsqueeze(dim=0)

            for step in range(self.num_steps):
                observation = torch.nn.functional.normalize(observation.to(self.device).unsqueeze(dim=0))
                for agent_id in self.env.agents:
                    with torch.no_grad():
                        value, action, action_log_prob = self.actor_critic_dict[agent_id].act(observation, full_log_prob=full_log_prob)

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

                masks = (~torch.tensor(done)).float().unsqueeze(0)

                #masks = 1 - mas_dict2tensor(done, int)
                rewards = mas_dict2tensor(rewards, int)
                actions = mas_dict2tensor(action_dict, int)
                values = mas_dict2tensor(values_dict, float)
                action_log_probs = mas_dict2tensor(action_log_dict, float if not full_log_prob else list)

                rollout.insert(
                    step=step,
                    state=observation.squeeze(dim=0),
                    action=actions,
                    values=values,
                    reward=rewards,
                    mask=masks,
                    action_log_probs=action_log_probs,
                )

                # update observation
                observation = new_observation

            ## fixme: qui bisogna capire il get_value a cosa serve e come farlo per multi agent
            with torch.no_grad():
                next_value = self.actor_critic_dict["agent_0"].get_value(rollout.states[-1].unsqueeze(0)).detach()

            rollout.compute_returns(next_value, True, self.gamma, 0.95)
            value_loss, action_loss, entropy = self.agent.update(rollout)

        return value_loss, action_loss, entropy, rollout

    def set_env(self, env):
        self.env = env

    def act(self, obs, agent_id, full_log_prob=False):
        return self.actor_critic_dict[agent_id].act(obs, deterministic=True, full_log_prob=True)


