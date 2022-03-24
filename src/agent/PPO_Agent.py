import torch
from torch import optim, nn


class PPO_Agent:
    def __init__(self, model, device, lr, eps, clip_param, clip_value_loss,
                 max_grad_norm, entropy_coef, value_loss_coef):

        self.actor_critic = model.to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)

        self.clip_param = clip_param
        self.clip_value_loss = clip_value_loss
        self.max_grad_norm = max_grad_norm

        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

    def act(self, observation, mask):
        return self.actor_critic.act(observation, mask)

    def get_value(self, observation, mask):
        return self.actor_critic.get_value(observation, mask)

    def get_modules(self):
        return self.actor_critic.get_modules()

    def eval(self):
        self.actor_critic.eval()

    def train(self):
        self.actor_critic.train()

    def ppo_step(self, states, actions, log_probs, values, returns, adv_targ, masks):
        def mean_fn(tensor): return float(tensor.mean())

        logs = dict(
            ratio=[], surr1=[], surr2=[], returns=[],
            adv_targ=[], perc_surr1=[], perc_surr2=[],
            curr_log_probs=[], old_log_probs=[], em_out=[],
        )

        curr_values, curr_log_probs, entropy, em_out = self.actor_critic.evaluate_actions(states, masks)

        logs["curr_log_probs"].append(mean_fn(curr_log_probs))
        logs["old_log_probs"].append(mean_fn(log_probs))
        logs["returns"].append(mean_fn(returns))
        logs["adv_targ"].append(mean_fn(values))
        #logs['em_out'].append(em_out)

        single_log_prob = log_probs.gather(-1, actions)
        single_curr_log_prob = curr_log_probs.gather(-1, actions)

        ratio = torch.exp(single_curr_log_prob - single_log_prob)
        surr1 = ratio * adv_targ
        surr2 = (
            torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            * adv_targ
        )

        action_loss = -torch.min(surr1, surr2).mean()
        logs["ratio"].append(mean_fn(ratio))
        logs["surr1"].append(mean_fn(surr1))
        logs["surr2"].append(mean_fn(surr2))
        logs["perc_surr1"].append(mean_fn((surr1 <= surr2).float()))
        logs["perc_surr2"].append(mean_fn((surr1 < surr2).float()))

        if self.clip_value_loss:
            value_pred_clipped = values + (curr_values - values).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (curr_values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            value_loss = (
                0.5 * torch.max(value_losses, value_losses_clipped).mean()
            )
        else:
            value_loss = 0.5 * (returns - curr_values).pow(2).mean()

        self.optimizer.zero_grad()

        value_loss *= self.value_loss_coef
        entropy *= self.entropy_coef
        loss = (
            value_loss
            + action_loss
            - entropy
        )
        loss.backward()

        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )
        self.optimizer.step()

        return action_loss, value_loss, entropy, logs
