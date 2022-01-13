import torch
import torch.nn as nn
from torch import optim


class PPO:
    def __init__(
            self,
            actor_critic_dict,
            clip_param,
            num_minibatch,
            value_loss_coef,
            entropy_coef,
            lr,
            eps,
            max_grad_norm,
            use_clipped_value_loss=True,
    ):

        self.actor_critic_dict = actor_critic_dict

        self.clip_param = clip_param
        self.num_minibatch = num_minibatch

        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizers = {
            agent_id: optim.Adam(
                self.actor_critic_dict[agent_id].parameters(), lr=lr, eps=eps
            )
            for agent_id in self.actor_critic_dict.keys()
        }

    def update(self, rollout, logs):
        advantages = rollout.returns[:-1] - rollout.values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)

        value_losses = torch.zeros(len(self.actor_critic_dict))
        action_losses = torch.zeros(len(self.actor_critic_dict))
        entropies = torch.zeros(len(self.actor_critic_dict))

        log_fn=lambda tensor: float(tensor.mean())

        data_generator = rollout.recurrent_generator(
            advantages, minibatch_frames=self.num_minibatch
        )

        for sample in data_generator:
            states_batch, recurrent_hs_batch, actions_batch, old_logs_probs_batch, \
            values_batch, return_batch, masks_batch, adv_targ = sample

            for agent_id in self.actor_critic_dict.keys():
                agent_index = int(agent_id[-1])  ## fixme: per il momento fatto così per l'indice

                agent_recurrent_hs = recurrent_hs_batch[:, agent_index]
                agent_log_probs = old_logs_probs_batch[:, agent_index, :]
                agent_actions = actions_batch[:, agent_index]
                agent_values = values_batch[:, agent_index]
                agent_returns = return_batch[:, agent_index]
                agent_adv_targ = adv_targ[:, agent_index]

                values, curr_log_porbs, entropy, _ = self.actor_critic_dict[agent_id].evaluate_actions(
                    states_batch, agent_recurrent_hs, masks_batch, agent_actions
                )

                ratio = torch.exp(curr_log_porbs - agent_log_probs)
                surr1 = ratio * agent_adv_targ
                surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                        * agent_adv_targ
                )

                logs[agent_id]["curr_log_porbs"].append(log_fn(curr_log_porbs))
                logs[agent_id]["old_log_probs"].append(log_fn(agent_log_probs))

                action_loss = torch.min(surr1, surr2)

                logs[agent_id]["perc_surr1"].append(log_fn((action_loss == surr1).float()))
                logs[agent_id]["perc_surr2"].append(log_fn((action_loss == surr2).float()))

                action_loss = -action_loss.mean()

                logs[agent_id]["ratio"].append(log_fn(ratio))
                logs[agent_id]["surr1"].append(log_fn(surr1))
                logs[agent_id]["surr2"].append(log_fn(surr2))
                logs[agent_id]["returns"].append(log_fn(agent_returns))
                logs[agent_id]["adv_targ"].append(log_fn(agent_adv_targ))

                if self.use_clipped_value_loss:
                    value_pred_clipped = agent_values + (values - agent_values).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (values - agent_returns).pow(2)
                    value_losses_clipped = (value_pred_clipped - agent_returns).pow(2)
                    value_loss = (
                            0.5 * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (agent_returns - values).pow(2).mean()

                self.optimizers[agent_id].zero_grad()

                value_loss *= self.value_loss_coef
                entropy *= self.entropy_coef
                loss = (
                        value_loss
                        + action_loss
                        - entropy
                )
                loss.backward()

                nn.utils.clip_grad_norm_(
                    self.actor_critic_dict[agent_id].parameters(), self.max_grad_norm
                )
                self.optimizers[agent_id].step()

                value_losses[agent_index] += value_loss.item()
                action_losses[agent_index] += action_loss.item()
                entropies[agent_index] += entropy.item()

        num_updates = max(rollout.step / self.num_minibatch, 1)

        value_losses /= num_updates
        action_losses /= num_updates
        entropies /= num_updates

        return value_losses.mean(), action_losses.mean(), entropies.mean()
