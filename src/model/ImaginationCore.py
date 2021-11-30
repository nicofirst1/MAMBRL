import torch
import torch.nn as nn

from src.model import EnvModel
from src.model.ModelFree import ModelFree


class ImaginationCore(nn.Module):
    def __init__(
        self,
        num_rollouts: int,
        in_shape,
        num_actions: int,
        num_rewards: int,
        env_model: EnvModel,
        model_free: ModelFree,
        device,
        num_frames: int,
        target2pix,
        full_rollout=True,
    ):
        super().__init__()
        self.num_rollouts = num_rollouts
        self.in_shape = in_shape
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.env_model = env_model
        self.model_free = model_free
        self.full_rollout = full_rollout
        self.device = device
        self.num_frames = num_frames
        self.target2pix = target2pix

    def forward(self, state):
        # state      = state.cpu()
        batch_size = state.size(0)

        rollout_states = []
        rollout_rewards = []

        if self.full_rollout:
            # esegui un rollout per ogni azione
            state = (
                state.unsqueeze(0)
                .repeat(self.num_actions, 1, 1, 1, 1)
                .view(-1, *self.in_shape)
            )
            action = torch.LongTensor(
                [[i] for i in range(self.num_actions)] * batch_size
            )
            action = action.view(-1)
            rollout_batch_size = batch_size * self.num_actions
        else:
            # get last state (discard num_frames)
            last_state = state[:, -self.in_shape[0] :, :]
            action = self.model_free.act(last_state)
            action = action.detach()
            rollout_batch_size = batch_size

        for step in range(self.num_rollouts):
            """
            propagate action on "image" (aka tensor of same shape as image)
            this tensor (onehot_action) has the same shape as the image on the last two dimension
            while the first one is based on the number of actions
            [batch size, actions , img_w, img_h]
            the index [:,idx,:,:] corresponding to the action is chosen and the image is set to 1, the others are zero
            """

            imagined_state, imagined_reward = self.env_model.full_pipeline(
                action, state
            )

            onehot_reward = torch.zeros(rollout_batch_size, self.num_rewards)
            onehot_reward[range(rollout_batch_size), imagined_reward] = 1

            # add a dimension for the rollout, then concat
            rollout_states.append(imagined_state.unsqueeze(0))
            rollout_rewards.append(onehot_reward.unsqueeze(0))

            # fix: we need 3 dimension when passing state to model free, hence
            # we squeeze it
            state = imagined_state.to(self.device)

            action = self.model_free.act(state)
            action = action.detach()

        rollout_states = torch.cat(rollout_states).to(self.device)
        rollout_rewards = torch.cat(rollout_rewards).to(self.device)

        return rollout_states, rollout_rewards
