import torch

class RolloutStorage(object):
    def __init__(self, num_steps, obs_shape, num_actions, num_agents):
        self.step = 0
        self.num_channels = obs_shape[0]
        self.num_actions = num_actions

        # todo: togli stati RNN
        self.states = torch.zeros(num_steps + 1, *obs_shape)
        #self.recurrent_hs = torch.zeros(num_steps + 1, num_agents, recurrent_hs_size)
        self.rewards = torch.zeros(num_steps, num_agents, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_agents, 1)
        self.returns = torch.zeros(num_steps + 1, num_agents, 1)
        self.actions = torch.zeros(num_steps, num_agents, 1).long()
        self.action_log_probs = torch.zeros(num_steps, num_agents, num_actions)
        self.masks = torch.ones(num_steps + 1, 1)

    def to(self, device):
        self.states = self.states.to(device)
        #self.recurrent_hs = self.recurrent_hs.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.actions = self.actions.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.masks = self.masks.to(device)

    def insert(self, state, action, action_log_probs, value_preds, reward, mask):
        self.states[self.step + 1].copy_(state)
        #self.recurrent_hs[self.step + 1].copy_(recurrent_hs)
        self.actions[self.step].copy_(action)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(reward)
        self.masks[self.step + 1].copy_(mask)

        self.step += 1

    def after_update(self):
        self.states[0].copy_(self.states[self.step])
        # self.recurrent_hs[0].copy_(self.recurrent_hs[self.step])
        # todo: commentedd mask copy cos is always zero
        # self.masks[0].copy_(self.masks[self.step])
        self.step = 0

    def compute_returns(self, next_value, use_gae, gamma, gae_lambda):
        if use_gae:
            # fixme: add self.step as index
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
        # todo: vedi se splittare i frames nel rollout oppure nell'env
        total_samples = self.rewards.size(0)
        perm = torch.randperm(total_samples)

        for start_ind in range(0, total_samples, minibatch_frames):
            states_minibatch = []
            #recurrent_hs_minibatch = []
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
                states_minibatch.append(self.states[ind].unsqueeze(dim=0))
                # recurrent_hs_minibatch.append(self.recurrent_hs[ind].unsqueeze(dim=0))
                actions_minibatch.append(self.actions[ind].unsqueeze(dim=0))
                log_probs_minibatch.append(
                    self.action_log_probs[ind].unsqueeze(dim=0))
                value_preds_minibatch.append(
                    self.value_preds[ind].unsqueeze(dim=0))
                return_minibatch.append(self.returns[ind].unsqueeze(dim=0))
                masks_minibatch.append(self.masks[ind])
                adv_targ_minibatch.append(advantages[ind].unsqueeze(dim=0))

            # cat on firt dimension
            states_minibatch = torch.cat(states_minibatch, dim=0)
            #recurrent_hs_minibatch = torch.cat(recurrent_hs_minibatch, dim=0)
            actions_minibatch = torch.cat(actions_minibatch, dim=0)
            value_preds_minibatch = torch.cat(value_preds_minibatch, dim=0)
            return_minibatch = torch.cat(return_minibatch, dim=0)
            masks_minibatch = torch.cat(masks_minibatch, dim=0)
            log_probs_minibatch = torch.cat(log_probs_minibatch, dim=0)
            adv_targ_minibatch = torch.cat(adv_targ_minibatch, dim=0)

            yield states_minibatch, actions_minibatch, log_probs_minibatch, \
                value_preds_minibatch, return_minibatch, masks_minibatch, adv_targ_minibatch