import torch


class RolloutStorage(object):
    def __init__(
        self, num_steps, state_shape, num_agents, gamma, size_minibatch, num_actions
    ):
        self.num_steps = num_steps
        self.num_channels = state_shape[0]
        self.num_agents = num_agents

        self.states = torch.zeros(num_steps + 1, *state_shape).long()
        self.rewards = torch.zeros(num_steps, num_agents).long()
        self.masks = torch.ones(num_steps + 1, num_agents).long()
        self.actions = torch.zeros(num_steps, num_agents).long()
        self.values = torch.zeros(num_steps + 1, num_agents).long()
        self.returns = torch.zeros(num_steps, num_agents)
        self.gae = torch.zeros(num_steps+1, num_agents)
        self.action_log_probs = torch.zeros(num_steps, num_agents)
        self.gamma = gamma
        self.size_minibatch = size_minibatch

    def to(self, device):
        self.states = self.states.to(device)
        self.rewards = self.rewards.to(device)
        self.masks = self.masks.to(device)
        self.actions = self.actions.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)

    def insert(self, step, state, action, values, reward, mask, action_log_probs):
        self.states[step + 1].copy_(state)
        self.actions[step].copy_(action)
        self.values[step].copy_(values)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)
        self.action_log_probs[step].copy_(action_log_probs)

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, tau=0.95):
        """compute_returns method.

        compute both the return (Q-Value) and the generalizaed advantage
        estimator https://arxiv.org/pdf/1506.02438.pdf by starting from the
        last timestep and going backward.
        """
        self.gae[-1] = 0
        self.values[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            delta = (
                self.rewards[step] +
                self.gamma * self.values[step + 1] * self.masks[step] -
                self.values[step]
            )
            self.gae[step] = delta + self.gamma * \
                tau * self.masks[step] * self.gae[step+1]
            # Q-value function (Advantage + Value)
            self.returns[step] = self.gae[step] + self.values[step]

    def get_num_minibatches(self):
        """get_num_minibatches method.

        Returns the number of possible minibatches based on the number of
        samples and the size of one minibatch
        """
        total_samples = self.rewards.size(0) - 1

        return total_samples // self.size_minibatch

    def recurrent_generator(self):
        """recurrent_generator method.

        Generates a set of permuted minibatches (use the get_num_minibatches
        method to obtain the exact number) of size self.size_minibatch

        Returns
        -------
        states_minibatch : torch.Tensor
            [minibatch_size, channels, width, height]
        actions_minibatch: torch.Tensor
            [minibatch_size, num_agents]
        return_minibatch: torch.Tensor
            [minibatch_size, num_agents]
        masks_minibatch: torch.Tensor
            [minibatch_size, num_agents]
        old_action_log_probs_minibatch: torch.Tensor
            [minibatch_size, num_agents]
        adv_targ_minibatch: torch.Tensor
            [minibatch_size, num_agents]
        next_states_minibatch: torch.Tensor
            [minibatch_size, num_channels, width, height]
        """
        total_samples = self.rewards.size(0) - 1
        perm = torch.randperm(total_samples)
        minibatch_frames = self.size_minibatch
        done = False

        for start_ind in range(0, total_samples, minibatch_frames):
            next_states_minibatch = []
            actions_minibatch = []
            return_minibatch = []
            masks_minibatch = []
            old_action_log_probs_minibatch = []
            adv_targ_minibatch = []
            states_minibatch = []

            for offset in range(minibatch_frames):
                if start_ind + minibatch_frames >= total_samples:
                    # skip last batch if not divisible
                    done = True
                    continue

                ind = perm[start_ind + offset]
                states_minibatch.append(self.states[ind].unsqueeze(0))
                next_states_minibatch.append(self.states[ind+1].unsqueeze(0))
                return_minibatch.append(self.returns[ind].unsqueeze(0))
                masks_minibatch.append(self.masks[ind].unsqueeze(0))
                actions_minibatch.append(self.actions[ind].unsqueeze(0))
                old_action_log_probs_minibatch.append(
                    self.action_log_probs[ind].unsqueeze(0)
                )
                adv_targ_minibatch.append(self.gae[ind].unsqueeze(0))

            if done:
                break

            # cat on firt dimension
            states_minibatch = torch.cat(states_minibatch, dim=0)
            next_states_minibatch = torch.cat(next_states_minibatch, dim=0)
            actions_minibatch = torch.cat(actions_minibatch, dim=0)
            return_minibatch = torch.cat(return_minibatch, dim=0)
            masks_minibatch = torch.cat(masks_minibatch, dim=0)
            old_action_log_probs_minibatch = torch.cat(
                old_action_log_probs_minibatch, dim=0)
            adv_targ_minibatch = torch.cat(adv_targ_minibatch, dim=0)

            yield states_minibatch, actions_minibatch, return_minibatch,\
                masks_minibatch, old_action_log_probs_minibatch,\
                adv_targ_minibatch, next_states_minibatch
