import torch


class RolloutStorage(object):
    def __init__(self, num_steps, state_shape, num_agents, gamma):
        self.num_steps = num_steps
        self.states = torch.zeros(num_steps + 1, *state_shape)
        self.rewards = torch.zeros(num_steps, num_agents)
        self.masks = torch.ones(num_steps + 1, num_agents)
        self.actions = torch.zeros(num_steps, num_agents).long()
        self.values = torch.zeros(num_steps + 1, num_agents).long()
        self.returns = torch.zeros(num_steps + 1, num_agents)
        self.action_log_probs = torch.zeros(num_steps, num_agents)
        self.gamma = gamma

    def to(self, device):
        self.states = self.states.to(device)
        self.rewards = self.rewards.to(device)
        self.masks = self.masks.to(device)
        self.actions = self.actions.to(device)

    def insert(self, step, state, action, values, reward, mask, action_log_probs):

        self.states[step + 1].copy_(torch.as_tensor(state))
        self.actions[step].copy_(torch.as_tensor(action))
        self.values[step].copy_(torch.as_tensor(values))
        self.rewards[step].copy_(torch.as_tensor(reward))
        self.masks[step + 1].copy_(torch.as_tensor(mask))

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, tau=0.95):
        self.values[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = self.rewards[step] + self.gamma * self.values[step + 1] * self.masks[step + 1] - self.values[
                step]
            gae = delta + self.gamma * tau * self.masks[step + 1] * gae
            self.returns[step] = gae + self.values[step]

    def recurrent_generator(self, advantages, num_mini_batch):
        total_samples = self.rewards.size(0)
        num_agents_per_batch = total_samples // num_mini_batch
        perm = torch.randperm(total_samples)
        for start_ind in range(0, total_samples, num_agents_per_batch):
            states_batch = []
            actions_batch = []
            return_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_agents_per_batch):
                ind = perm[start_ind + offset]
                states_batch.append(self.states[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            states_batch = torch.cat(states_batch, 0)
            actions_batch = torch.cat(actions_batch, 0)
            return_batch = torch.cat(return_batch, 0)
            masks_batch = torch.cat(masks_batch, 0)
            old_action_log_probs_batch = torch.cat(old_action_log_probs_batch, 0)
            adv_targ = torch.cat(adv_targ, 0)

            yield states_batch, actions_batch, \
                  return_batch, masks_batch, old_action_log_probs_batch, adv_targ
