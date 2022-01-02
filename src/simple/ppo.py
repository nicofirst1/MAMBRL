import torch
import torch.nn as nn
from torch import optim


class PPO:
    def __init__(self, actor_critic_dict, clip_param, ppo_epoch, num_minibatch, value_loss_coef,
            entropy_coef, lr, eps, max_grad_norm, use_clipped_value_loss=True):

        self.actor_critic_dict = actor_critic_dict

        self.ppo_epoch = ppo_epoch
        self.clip_param = clip_param
        self.num_minibatch = num_minibatch

        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizers = {
            agent_id: optim.Adam(self.actor_critic_dict[agent_id].parameters(), lr=lr, eps=eps)
            for agent_id in self.actor_critic_dict.keys()
        }

    def update(self, rollout):
        advantages = rollout.returns[:-1] - rollout.values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        ## fixme: serve questo ciclo? Noi non lo avevamo
        for e in range(self.ppo_epoch):
            data_generator = rollout.recurrent_generator(advantages)

            for sample in data_generator:
                states_batch, actions_batch, values_batch, return_batch, \
                masks_batch, old_action_logs_probs_batch, adv_targ, _, = sample

                for agent_id in self.actor_critic_dict.keys():
                    agent_index = int(agent_id[-1]) ## fixme: per il momento fatto così per l'indice

                    agent_actions = actions_batch[:, agent_index].unsqueeze(dim=-1)
                    agent_values = values_batch[:, agent_index].unsqueeze(dim=-1)
                    agent_returns = return_batch[:, agent_index].unsqueeze(dim=-1)
                    agent_action_log_probs = old_action_logs_probs_batch[:, agent_index].unsqueeze(dim=-1)
                    agent_adv_targ = adv_targ[:, agent_index].unsqueeze(dim=-1)

                    values, actions_log_probs, entropy = self.actor_critic_dict[agent_id].evaluate_actions(states_batch, agent_actions)

                    ratio = torch.exp(actions_log_probs - agent_action_log_probs)
                    surr1 = ratio * agent_adv_targ
                    surr2 = (torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * agent_adv_targ)

                    action_loss = -torch.min(surr1, surr2).mean()

                    if self.use_clipped_value_loss:
                        value_pred_clipped = agent_values + (values - agent_values).clamp(-self.clip_param, self.clip_param)
                        value_losses = (values - agent_returns).pow(2)
                        value_losses_clipped = (value_pred_clipped - agent_returns).pow(2)
                        value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    else:
                        value_loss = 0.5 * (agent_returns - values).pow(2).mean()

                    self.optimizers[agent_id].zero_grad()

                    loss = value_loss * self.value_loss_coef + action_loss - entropy * self.entropy_coef
                    loss.backward()

                    nn.utils.clip_grad_norm_(self.actor_critic_dict[agent_id].parameters(), self.max_grad_norm)
                    self.optimizers[agent_id].step()

                    value_loss_epoch += value_loss.item()
                    action_loss_epoch += action_loss.item()
                    dist_entropy_epoch += entropy.item()

        num_updates = self.ppo_epoch * self.num_minibatch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
