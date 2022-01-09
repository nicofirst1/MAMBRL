import torch


class RolloutStorage(object):
    def __init__(self, num_steps, obs_shape, num_agents, num_actions):
        self.steps = 0
        self.num_channels = obs_shape[0]

        self.states = torch.zeros(num_steps + 1, *obs_shape)
        # self.next_states = torch.zeros(num_steps, *state_shape)
        self.rewards = torch.zeros(num_steps, num_agents, 1)
        self.masks = torch.ones(num_steps + 1, num_agents, 1)
        self.actions = torch.zeros(num_steps, num_agents, 1).long()
        self.values = torch.zeros(num_steps + 1, num_agents, 1)
        self.returns = torch.zeros(num_steps + 1, num_agents, 1)
        self.action_log_probs = torch.zeros(num_steps, num_agents, num_actions)

    def to(self, device):
        self.states = self.states.to(device)
        self.rewards = self.rewards.to(device)
        self.masks = self.masks.to(device)
        self.actions = self.actions.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)

    def insert(self, step, state, action, values, reward, mask, action_log_probs):
        self.states[self.steps + 1].copy_(state)
        self.actions[self.steps].copy_(action)
        self.values[self.steps].copy_(values)
        self.rewards[self.steps].copy_(reward)
        self.masks[self.steps + 1].copy_(mask)
        self.action_log_probs[step].copy_(action_log_probs)

        self.steps += 1

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda):
        if use_gae:
            self.values[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.values[step + 1] * self.masks[step + 1]
                    - self.values[step]
                )
                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.values[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def recurrent_generator(self, advantages, minibatch_frames):
        """
        Generates a set of permuted minibatches (use the get_num_minibatches
        method to obtain the exact number) of size self.size_minibatch

        Returns:
            states_minibatch : torch.Tensor[minibatch_size, channels, width, height]
            actions_minibatch: torch.Tensor[minibatch_size, num_agents]
            return_minibatch: torch.Tensor[minibatch_size, num_agents]
            masks_minibatch: torch.Tensor[minibatch_size, num_agents]
            old_action_log_probs_minibatch: torch.Tensor[minibatch_size, num_agents]
            adv_targ_minibatch: torch.Tensor[minibatch_size, num_agents]
            next_states_minibatch: torch.Tensor[minibatch_size, num_channels, width, height]
        """
        #total_samples = self.rewards.size(0)
        total_samples = self.steps
        perm = torch.randperm(total_samples)
        done = False

        if minibatch_frames >= total_samples:
            minibatch_frames = total_samples

        for start_ind in range(0, total_samples, minibatch_frames):
            next_states_minibatch = []
            actions_minibatch = []
            values_minibatch = []
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
                # next_states_minibatch.append(self.next_states[ind].unsqueeze(0))
                actions_minibatch.append(self.actions[ind].unsqueeze(0))
                values_minibatch.append(self.values[ind].unsqueeze(0))
                return_minibatch.append(self.returns[ind].unsqueeze(0))
                masks_minibatch.append(self.masks[ind].unsqueeze(0))
                old_action_log_probs_minibatch.append(self.action_log_probs[ind].unsqueeze(0))
                adv_targ_minibatch.append(advantages[ind].unsqueeze(0))

            if done:
                break

            # cat on firt dimension
            states_minibatch = torch.cat(states_minibatch, dim=0)
            # next_states_minibatch = torch.cat(next_states_minibatch, dim=0)
            actions_minibatch = torch.cat(actions_minibatch, dim=0)
            values_minibatch = torch.cat(values_minibatch, dim=0)
            return_minibatch = torch.cat(return_minibatch, dim=0)
            masks_minibatch = torch.cat(masks_minibatch, dim=0)
            old_action_log_probs_minibatch = torch.cat(
                old_action_log_probs_minibatch, dim=0
            )
            adv_targ_minibatch = torch.cat(adv_targ_minibatch, dim=0)

            yield states_minibatch, actions_minibatch, values_minibatch, return_minibatch, masks_minibatch, old_action_log_probs_minibatch, adv_targ_minibatch, next_states_minibatch
