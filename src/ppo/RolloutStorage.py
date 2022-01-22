import torch


class RolloutStorage_francesco(object):
    def __init__(
            self, num_steps, state_shape, num_agents, num_actions, gamma, size_minibatch
    ):
        self.num_steps = num_steps
        self.num_channels = state_shape[0]
        self.num_agents = num_agents
        self.gamma = gamma
        self.size_minibatch = size_minibatch
        self.num_actions = num_actions

        # states are not normalized
        self.states = torch.zeros(num_steps, *state_shape)
        self.next_states = torch.zeros(num_steps, *state_shape)
        self.rewards = torch.zeros(num_steps, num_agents)
        self.masks = torch.ones(num_steps, num_agents).long()
        self.actions = torch.zeros(num_steps, num_agents).long()
        self.values = torch.zeros(num_steps, num_agents)
        self.returns = torch.zeros(num_steps, num_agents)
        self.gae = torch.zeros(num_steps, num_agents)
        self.action_log_probs = torch.zeros(num_steps, num_agents, num_actions)

    def to(self, device):
        self.states = self.states.to(device)
        self.rewards = self.rewards.to(device)
        self.masks = self.masks.to(device)
        self.actions = self.actions.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.values = self.values.to(device)
        self.returns = self.returns.to(device)
        self.gae = self.gae.to(device)

    def insert(self, step, state, next_state, action, values, reward, mask, action_log_probs):
        self.states[step].copy_(state)
        self.next_states[step].copy_(next_state)
        self.actions[step].copy_(action)
        self.values[step].copy_(values)
        self.rewards[step].copy_(reward)
        self.masks[step].copy_(mask)
        self.action_log_probs[step].copy_(action_log_probs)

    def after_update(self):
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, tau=0.95):
        """compute_returns method.

        compute both the return (Q-Value) and the generalizaed advantage
        estimator https://arxiv.org/pdf/1506.02438.pdf by starting from the
        last timestep and going backward.
        Parameters
        ----------
        tau: float
            parameter used for the variance/bias tradeoff tau = 1 high variance
            tau = 0 high bias
        """
        # self.gae[-1] = 0
        # self.values[-1] = next_value
        # for step in reversed(range(self.rewards.size(0))):
        #     delta = (
        #         self.rewards[step] +
        #         self.gamma * self.values[step + 1] * self.masks[step+1] -
        #         self.values[step]
        #     )
        # We use this version in order to use the same size for all tensors
        for step in reversed(range(self.rewards.size(0))):
            if step == self.rewards.size(0) - 1:
                delta = self.rewards[step] - self.values[step]
                self.gae[step] = delta
            else:
                delta = (
                        self.rewards[step] +
                        self.gamma * self.values[step + 1] * self.masks[step] -
                        self.values[step]
                )
                self.gae[step] = delta + self.gamma * \
                                 tau * self.masks[step] * self.gae[step + 1]

            # fix: advantage normalization, not sure if we should use it or not
            # self.gae = (self.gae - self.gae.mean()) / (self.gae.std() + 1e-5)
            # return = advantage estimator
            # https://github.com/higgsfield/RL-Adventure-2/issues/6
            self.returns[step] = self.gae[step] + self.values[step]

    def get_num_minibatches(self):
        """get_num_minibatches method.

        Returns the number of possible minibatches based on the number of
        samples and the size of one minibatch
        """
        total_samples = self.rewards.size(0)

        return total_samples // self.size_minibatch

    def recurrent_generator(self, num_frames=0):
        """recurrent_generator method.

        Generates a set of permuted minibatches (use the get_num_minibatches
        method to obtain the exact number) of size self.size_minibatch.
        states and next_states are returned normalized between 0 and 1
        Parameters
        ----------
        num_frames: int
            frames to stack together. The generator take a current frame and
            the previous num_frames. If the frames are the intial frames, it
            copies the same frames multiple times
        Returns
        -------
        states_minibatch : torch.Tensor
            [minibatch_size, channels, width, height] if num_frames == 0 else
            [minibatch_size, channels, num_frames, width, height]
        actions_minibatch: torch.Tensor
            [minibatch_size, num_agents] if num_frames == 0 else
        return_minibatch: torch.Tensor
            [minibatch_size, num_agents]
        masks_minibatch: torch.Tensor
            [minibatch_size, num_agents]
        old_action_log_probs_minibatch: torch.Tensor
            [minibatch_size, num_agents]
        adv_targ_minibatch: torch.Tensor
            [minibatch_size, num_agents]
        next_states_minibatch: torch.Tensor
            [minibatch_size, num_channels, width, height] if num_frames == 0
            else [minibatch_size, num_channels, width, height]
        """
        total_samples = self.rewards.size(0)
        perm = torch.randperm(total_samples)
        minibatch_frames = self.size_minibatch
        done = False

        if num_frames == 0:
            for start_ind in range(0, total_samples, minibatch_frames):
                next_states_minibatch = []
                actions_minibatch = []
                return_minibatch = []
                value_minibatch = []
                masks_minibatch = []
                old_action_log_probs_minibatch = []
                rewards_minibatch = []
                adv_targ_minibatch = []
                states_minibatch = []

                for offset in range(minibatch_frames):
                    if start_ind + minibatch_frames >= total_samples:
                        # skip last batch if not divisible
                        done = True
                        continue

                    ind = perm[start_ind + offset]
                    states_minibatch.append(
                        (self.states[ind] / 255.).unsqueeze(0))
                    next_states_minibatch.append(
                        (self.next_states[ind] / 255.).unsqueeze(0))
                    value_minibatch.append(self.values[ind].unsqueeze(0))
                    return_minibatch.append(self.returns[ind].unsqueeze(0))
                    masks_minibatch.append(self.masks[ind].unsqueeze(0))
                    rewards_minibatch.append(self.rewards[ind].unsqueeze(0))
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
                value_minibatch = torch.cat(value_minibatch, dim=0)
                masks_minibatch = torch.cat(masks_minibatch, dim=0)
                rewards_minibatch = torch.cat(rewards_minibatch, dim=0)
                old_action_log_probs_minibatch = torch.cat(
                    old_action_log_probs_minibatch, dim=0)
                adv_targ_minibatch = torch.cat(adv_targ_minibatch, dim=0)

                yield states_minibatch, actions_minibatch, value_minibatch, \
                      return_minibatch, rewards_minibatch, \
                      masks_minibatch, old_action_log_probs_minibatch, \
                      adv_targ_minibatch, next_states_minibatch
        else:
            for start_ind in range(0, total_samples, minibatch_frames):
                next_states_minibatch = []
                actions_minibatch = []
                return_minibatch = []
                value_minibatch = []
                masks_minibatch = []
                old_action_log_probs_minibatch = []
                rewards_minibatch = []
                adv_targ_minibatch = []
                states_minibatch = []

                for offset in range(minibatch_frames):
                    if start_ind + minibatch_frames >= total_samples:
                        # skip last batch if not divisible
                        done = True
                        continue

                    ind = perm[start_ind + offset]
                    temp_states = []

                    # check if there are enough previous frames
                    if (ind + 1) - num_frames >= 0:
                        for i in range((ind + 1) - num_frames, ind + 1):
                            temp_states.append(
                                (self.states[i] / 255.).unsqueeze(1))
                    else:
                        for i in range(num_frames):
                            temp_states.append(
                                (self.states[ind] / 255.).unsqueeze(1))

                    states_minibatch.append(
                        torch.cat(temp_states, dim=1).unsqueeze(0))

                    next_states_minibatch.append(
                        (self.next_states[ind] / 255.).unsqueeze(0))
                    value_minibatch.append(self.values[ind].unsqueeze(0))
                    return_minibatch.append(self.returns[ind].unsqueeze(0))
                    masks_minibatch.append(self.masks[ind].unsqueeze(0))
                    rewards_minibatch.append(self.rewards[ind].unsqueeze(0))
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
                value_minibatch = torch.cat(value_minibatch, dim=0)
                masks_minibatch = torch.cat(masks_minibatch, dim=0)
                rewards_minibatch = torch.cat(rewards_minibatch, dim=0)
                old_action_log_probs_minibatch = torch.cat(
                    old_action_log_probs_minibatch, dim=0)
                adv_targ_minibatch = torch.cat(adv_targ_minibatch, dim=0)

                yield states_minibatch, actions_minibatch, value_minibatch, \
                      return_minibatch, rewards_minibatch, \
                      masks_minibatch, old_action_log_probs_minibatch, \
                      adv_targ_minibatch, next_states_minibatch


class RolloutStorage(object):
    def __init__(self, num_steps, obs_shape, num_agents, recurrent_hs_size):
        self.step = 0
        self.num_channels = obs_shape[0]

        #todo: togli stati RNN
        self.states = torch.zeros(num_steps + 1, *obs_shape)
        self.recurrent_hs = torch.zeros(num_steps + 1, num_agents, recurrent_hs_size)
        self.rewards = torch.zeros(num_steps, num_agents, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_agents, 1)
        self.returns = torch.zeros(num_steps + 1, num_agents, 1)
        self.actions = torch.zeros(num_steps, num_agents, 1).long()
        self.action_log_probs = torch.zeros(num_steps, num_agents, 1)
        self.masks = torch.ones(num_steps + 1, 1)

    def to(self, device):
        self.states = self.states.to(device)
        self.recurrent_hs = self.recurrent_hs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.masks = self.masks.to(device)

    def insert(self, state, recurrent_hs, action, action_log_probs, value_preds, reward, mask):
        self.states[self.step + 1].copy_(state)
        self.recurrent_hs[self.step + 1].copy_(recurrent_hs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step += 1

    def after_update(self):
        self.states[0].copy_(self.states[self.step])
        self.recurrent_hs[0].copy_(self.recurrent_hs[self.step])
        self.masks[0].copy_(self.masks[self.step])
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda):
        if use_gae:
            #fixme: add self.step as index
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (self.rewards[step] + gamma * self.value_preds[step + 1]
                         * self.masks[step + 1] - self.value_preds[step])

                gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = self.returns[step + 1] * \
                    gamma * self.masks[step + 1] + self.rewards[step]

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
        #todo: vedi se splittare i frames nel rollout oppure nell'env
        total_samples = self.rewards.size(0)
        perm = torch.randperm(total_samples)

        for start_ind in range(0, total_samples, minibatch_frames):
            states_minibatch = []
            recurrent_hs_minibatch = []
            actions_minibatch = []
            log_probs_minibatch = []
            value_preds_minibatch = []
            return_minibatch = []
            masks_minibatch = []
            adv_targ_minibatch = []

            if start_ind + minibatch_frames > total_samples:
                minibatch_frames = total_samples - start_ind

            for offset in range(minibatch_frames):
                ind = perm[start_ind + offset]
                states_minibatch.append(self.states[ind])
                recurrent_hs_minibatch.append(self.recurrent_hs[ind])
                actions_minibatch.append(self.actions[ind])
                log_probs_minibatch.append(self.action_log_probs[ind])
                value_preds_minibatch.append(self.value_preds[ind])
                return_minibatch.append(self.returns[ind])
                masks_minibatch.append(self.masks[ind])
                adv_targ_minibatch.append(advantages[ind])

            # cat on firt dimension
            states_minibatch = torch.cat(states_minibatch, dim=0)
            recurrent_hs_minibatch = torch.cat(recurrent_hs_minibatch, dim=0)
            actions_minibatch = torch.cat(actions_minibatch, dim=0)
            value_preds_minibatch = torch.cat(value_preds_minibatch, dim=0)
            return_minibatch = torch.cat(return_minibatch, dim=0)
            masks_minibatch = torch.cat(masks_minibatch, dim=0)
            log_probs_minibatch = torch.cat(log_probs_minibatch, dim=0)
            adv_targ_minibatch = torch.cat(adv_targ_minibatch, dim=0)


            yield states_minibatch, recurrent_hs_minibatch, actions_minibatch, log_probs_minibatch, \
                  value_preds_minibatch, return_minibatch, masks_minibatch, adv_targ_minibatch
