import math

import torch
import torch.nn as nn
from torch import optim
from .RolloutStorage import RolloutStorage
from typing import Dict


class PPO:
    def __init__(
            self,
            actor_critic_dict,
            ppo_epochs,
            clip_param,
            num_minibatch,
            value_loss_coef,
            entropy_coef,
            lr,
            eps,
            max_grad_norm,
            use_clipped_value_loss=True,
    ):
        """__init__ method.

        create a PPO agent. A PPO agent must implement the update function
        which, given a rollout, takes care of updating the parameters
        of its model
        Parameters
        ----------
        actor_critic_dict : Dict[str, ModelFree]
        ppo_epochs : int
            number of times the PPO update its model's parameters over the
            same set of trajectories
        clip_param : float
        num_minibatch : int
            number of elements inside a minibatch
        value_loss_coef : float
        entropy_coef : float
        lr : float
        eps : float
        max_grad_norm : int
        use_clipped_value_loss : Bool, optional
            The default is True.

        """

        self.actor_critic_dict = actor_critic_dict
        self.ppo_epochs = ppo_epochs

        self.clip_param = clip_param
        self.num_minibatch = num_minibatch

        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizers = {
            agent_id: optim.Adam(
                self.actor_critic_dict[agent_id].get_all_parameters(), lr=lr, eps=eps
            )

            for agent_id in self.actor_critic_dict.keys()
        }

    def train(self):
        for model in self.actor_critic_dict.values():
            model.train()

    def eval(self):
        for model in self.actor_critic_dict.values():
            model.eval()

    def update(self, rollout: RolloutStorage, logs: Dict[str, Dict[str, list]]):
        """update method.

        update the models parameters inside the self.actor_critic_dict
        given a rollout
        Parameters
        ----------
        rollout : RolloutStorage
        logs : logs: Dict[str,Dict[str,list]]
            update the log of each agents

        Returns
        -------
        value_loss: float
        action_loss: float
        entropy_loss: float

        """
        advantages = rollout.returns[:-1] - rollout.value_preds[:-1]
        # advantages = (advantages - advantages.mean()) / \
        #     (advantages.std() + 1e-10)

        agents_value_losses = torch.zeros(len(self.actor_critic_dict))
        agents_action_losses = torch.zeros(len(self.actor_critic_dict))
        agents_entropies = torch.zeros(len(self.actor_critic_dict))

        def mean_fn(tensor): return float(tensor.mean())

        for _ in range(self.ppo_epochs):
            data_generator = rollout.recurrent_generator(
                advantages, minibatch_frames=self.num_minibatch
            )

            for sample in data_generator:
                states_batch, actions_batch, old_logs_probs_batch, \
                    values_batch, return_batch, masks_batch, adv_targ = sample

                for agent_id in self.actor_critic_dict.keys():
                    # fixme: per il momento fatto così per l'indice
                    agent_index = int(agent_id[-1])

                    #agent_recurrent_hs = recurrent_hs_batch[:, agent_index]
                    old_log_probs = old_logs_probs_batch[:, agent_index, :]
                    agent_actions = actions_batch[:, agent_index]
                    agent_values = values_batch[:, agent_index]
                    agent_returns = return_batch[:, agent_index]
                    agent_adv_targ = adv_targ[:, agent_index]

                    # FIXED: NORMALIZE THE STATE
                    values, curr_log_probs, entropy = self.actor_critic_dict[agent_id].evaluate_actions(
                        states_batch, masks_batch, agent_actions
                    )

                    logs[agent_id]["curr_log_probs"].append(mean_fn(curr_log_probs))
                    logs[agent_id]["old_log_probs"].append(mean_fn(old_log_probs))
                    logs[agent_id]["returns"].append(mean_fn(agent_returns))
                    logs[agent_id]["adv_targ"].append(mean_fn(agent_adv_targ))

                    single_action_log_prob = curr_log_probs.gather(
                        -1, agent_actions)
                    single_action_old_log_prob = \
                        old_log_probs.gather(
                            -1, agent_actions)

                    ratio = torch.exp(single_action_log_prob -
                                      single_action_old_log_prob)
                    surr1 = ratio * agent_adv_targ
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param)
                        * agent_adv_targ
                    )

                    action_loss = -torch.min(surr1, surr2).mean()

                    logs[agent_id]["ratio"].append(mean_fn(ratio))
                    logs[agent_id]["surr1"].append(mean_fn(surr1))
                    logs[agent_id]["surr2"].append(mean_fn(surr2))
                    logs[agent_id]["perc_surr1"].append(mean_fn((surr1 <= surr2).float()))
                    logs[agent_id]["perc_surr2"].append(mean_fn((surr1 < surr2).float()))

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

                    # =============================================================================
                    # TO PRINT THE WHOLE COMPUTATIONAL GRAPH (FOR THE ACTOR, CRITIC AND BOTHS)
                    # =============================================================================
                    # getBack(loss.grad_fn)
                    # params_dict = dict(self.actor_critic_dict["agent_0"].named_parameters())
                    # make_dot(loss, params=params_dict,
                    #          show_attrs=True, show_saved=True).render("model_backward_graph", format="png")
                    # =============================================================================

                    nn.utils.clip_grad_norm_(
                        self.actor_critic_dict[agent_id].get_all_parameters(
                        ), self.max_grad_norm
                    )
                    self.optimizers[agent_id].step()

                    agents_value_losses[agent_index] += value_loss.item()
                    agents_action_losses[agent_index] += action_loss.item()
                    agents_entropies[agent_index] += entropy.item()

        num_updates = self.ppo_epochs * int(math.ceil(rollout.rewards.size(0) / self.num_minibatch))

        agents_value_losses /= num_updates
        agents_action_losses /= num_updates
        agents_entropies /= num_updates

        return agents_value_losses.mean(), agents_action_losses.mean(), agents_entropies.mean()
