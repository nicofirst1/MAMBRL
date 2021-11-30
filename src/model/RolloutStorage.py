import torch


class RolloutStorage(object):
    def __init__(
        self, num_steps, state_shape, num_agents, gamma, size_mini_batch, num_actions
    ):
        self.num_steps = num_steps
        self.state_shape = state_shape

        self.states = torch.zeros(num_steps + 1, *state_shape).long()
        self.rewards = torch.zeros(num_steps, num_agents).long()
        self.masks = torch.ones(num_steps + 1, num_agents).long()
        self.actions = torch.zeros(num_steps, num_agents).long()
        self.values = torch.zeros(num_steps + 1, num_agents).long()
        self.returns = torch.zeros(num_steps + 1, num_agents)
        self.action_log_probs = torch.zeros(num_steps + 1, num_actions, num_agents)
        self.gamma = gamma
        self.size_mini_batch = size_mini_batch

    def to(self, device):
        self.states = self.states.to(device)
        self.rewards = self.rewards.to(device)
        self.masks = self.masks.to(device)
        self.actions = self.actions.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)

    def insert(self, step, state, action, values, reward, mask, action_log_probs):
        self.states[step + 1].copy_(state[: self.state_shape[0]])
        self.actions[step].copy_(action)
        self.values[step].copy_(values)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)
        self.action_log_probs[step].copy_(action_log_probs)

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, tau=0.95):
        # todo: check if correct from formula

        self.values[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.size(0))):
            delta = (
                self.rewards[step]
                + self.gamma * self.values[step + 1] * self.masks[step + 1]
                - self.values[step]
            )
            gae = delta + self.gamma * tau * self.masks[step + 1] * gae
            self.returns[step] = gae + self.values[step]

    def get_num_batches(self, num_frames):
        total_samples = self.rewards.size(0) - 1
        minibatch_frames = self.size_mini_batch * num_frames

        return total_samples // minibatch_frames

    def recurrent_generator(self, advantages, num_frames):
        """recurrent_generator method."""
        total_samples = self.rewards.size(0) - 1
        # perm = torch.randperm(total_samples)

        minibatch_frames = self.size_mini_batch
        done = False

        obs_shape = (
            self.state_shape[0] * num_frames,
            self.state_shape[1],
            self.state_shape[2],
        )
        state_channel = self.state_shape[0]
        observation = torch.zeros(obs_shape).to(self.states.device)

        for start_ind in range(0, total_samples, minibatch_frames):
            states_batch = []
            actions_batch = []
            return_batch = []
            reward_batch = []
            masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(minibatch_frames):
                if start_ind + offset >= total_samples:
                    # skip last batch if not divisible
                    done = True
                    continue

                ind = start_ind + offset
                observation = torch.cat(
                    [observation[state_channel:, :, :], self.states[ind]], dim=0
                )

                states_batch.append(observation.unsqueeze(0))
                return_batch.append(self.returns[ind].unsqueeze(0))
                reward_batch.append(self.rewards[ind].unsqueeze(0))
                masks_batch.append(self.masks[ind].unsqueeze(0))
                actions_batch.append(self.actions[ind].unsqueeze(0))
                old_action_log_probs_batch.append(
                    self.action_log_probs[ind].unsqueeze(0)
                )
                adv_targ.append(advantages[ind].unsqueeze(0))

            if done:
                break

            # cat on firt dimension
            states_batch = torch.cat(states_batch, dim=0)
            actions_batch = torch.cat(actions_batch, dim=0)
            return_batch = torch.cat(return_batch, dim=0)
            reward_batch = torch.cat(reward_batch, dim=0)
            masks_batch = torch.cat(masks_batch, dim=0)
            old_action_log_probs_batch = torch.cat(old_action_log_probs_batch, dim=0)
            adv_targ = torch.cat(adv_targ, dim=0)

            # split per num frame
            # [batch * num_frames , *shape] ->[batch, num_frames , *shape]
            # states_batch = states_batch.view(
            #    -1, self.state_shape[0] * num_frames, *states_batch.shape[2:]
            # )
            # actions_batch = actions_batch.view(-1, num_frames, *actions_batch.shape[1:])
            # return_batch = return_batch.view(-1, num_frames, *return_batch.shape[1:])
            # masks_batch = masks_batch.view(-1, num_frames, *masks_batch.shape[1:])
            # old_action_log_probs_batch = old_action_log_probs_batch.view(
            #    -1, num_frames, *old_action_log_probs_batch.shape[1:]
            # )
            # adv_targ = adv_targ.view(-1, num_frames, *adv_targ.shape[1:])

            yield states_batch, actions_batch, return_batch, reward_batch, masks_batch, old_action_log_probs_batch, adv_targ

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])
