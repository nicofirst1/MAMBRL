import torch


class RolloutStorage(object):
    def __init__(self, num_steps, state_shape, num_agents, gamma):
        self.num_steps = num_steps
        self.states = torch.zeros(num_steps + 1, *state_shape)
        self.rewards = torch.zeros(num_steps, num_agents)
        self.masks = torch.ones(num_steps + 1, num_agents)
        self.actions = torch.zeros(num_steps, num_agents).long()
        self.values = torch.zeros(num_steps, num_agents).long()

        self.gamma= gamma

    def to(self,device):
        self.states = self.states.to(device)
        self.rewards = self.rewards.to(device)
        self.masks = self.masks.to(device)
        self.actions = self.actions.to(device)

    def insert(self, step, state, action, values,reward, mask):

        self.states[step + 1].copy_(torch.as_tensor(state))
        self.actions[step].copy_(torch.as_tensor(action))
        self.values[step].copy_(torch.as_tensor(values))
        self.rewards[step].copy_(torch.as_tensor(reward))
        self.masks[step + 1].copy_(torch.as_tensor(mask))

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        # fixme: che e' sta roba?
        for step in reversed(range(self.num_steps)):
            next_value[step] = next_value[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]
        return next_value[:-1]

