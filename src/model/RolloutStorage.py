import torch


class RolloutStorage(object):
    def __init__(self, num_steps, state_shape, num_agents):
        self.num_steps = num_steps
        self.states = torch.zeros(num_steps + 1, *state_shape)
        self.rewards = torch.zeros(num_steps, num_agents)
        self.masks = torch.ones(num_steps + 1, num_agents)
        self.actions = torch.zeros(num_steps, 1).long()
        self.use_cuda = False

    def cuda(self):
        self.use_cuda = True
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.masks = self.masks.cuda()
        self.actions = self.actions.cuda()

    def insert(self, step, state, action, reward, mask):

        self.states[step + 1].copy_(torch.as_tensor(state))
        self.actions[step].copy_(torch.as_tensor(action))
        self.rewards[step].copy_(torch.as_tensor(reward))
        self.masks[step + 1].copy_(torch.as_tensor(mask))

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma):

        for step in reversed(range(self.num_steps)):
            next_value[step] = next_value[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]
        return next_value[:-1]

