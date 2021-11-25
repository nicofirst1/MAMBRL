import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import EnvModel
from src.model.ModelFree import ModelFree

"""
    Here this has to be clarified.. what is this pixels and how the function
    target_to_pix is used!
"""


def target_to_pix(num_colors):
    color_index = [  # map index to RGB colors
        (0, 255, 0),  # green -> landmarks
        (0, 0, 255),  # blue -> agents
        (255, 255, 255),  # white -> background
    ]

    color_index = [torch.as_tensor(x) for x in color_index]

    """
        To pixel will become this:
            {0: (0.0, 1.0, 0.0), 1: (0.0, 1.0, 1.0), 2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0), 4: (1.0, 1.0, 0.0), 5: (0.0, 0.0, 0.0), 6: (1.0, 0.0, 0.0)}
    
        so this means that the imagined_states coming as input should have values only
        in the range 0 - 6. This is weird!
        N.B. with paras.out_shape = (3, 32, 32), the input imagined_states has shape (1024,)
    """

    assert num_colors == len(
        color_index), "The number of colors in param does not match colors in list"

    def inner(imagined_states):
        batch_size = imagined_states.shape[0]
        image_shape = imagined_states.shape[-2:]

        new_imagined_state = torch.zeros([batch_size, *image_shape, 3]).long()

        # remove channel dim since is 1
        imagined_states = imagined_states.squeeze(1)

        for c in range(num_colors):
            indices = imagined_states == c
            new_imagined_state[indices] = color_index[c]

        new_imagined_state = new_imagined_state.view(
            batch_size, 3, *image_shape)

        if False:  # debug, show image
            from PIL import Image
            img = new_imagined_state[0].cpu().view(32, 32, 3)
            img = Image.fromarray(img)
            img.show()

        return new_imagined_state

    return inner


class ImaginationCore(nn.Module):
    def __init__(
            self,
            num_rolouts: int,
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
        self.num_rolouts = num_rolouts
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
            last_state = state[:, -self.in_shape[0]:, :]
            action = self.model_free.act(last_state)
            action = action.detach()
            rollout_batch_size = batch_size

        for step in range(self.num_rolouts):
            """
            propagate action on "image" (aka tensor of same shape as image)
            this tensor (onehot_action) has the same shape as the image on the last two dimension
            while the first one is based on the number of actions
            [batch size, actions , img_w, img_h]
            the index [:,idx,:,:] corresponding to the action is chosen and the image is set to 1, the others are zero
            """
            onehot_action = torch.zeros(
                rollout_batch_size, self.num_actions, *self.in_shape[1:]
            ).to(self.device)
            onehot_action[range(rollout_batch_size), action] = 1
            inputs = torch.cat([state, onehot_action], 1).to(self.device)

            imagined_state, imagined_reward = self.env_model(inputs)

            imagined_state = F.softmax(imagined_state, dim=1).max(dim=1)[1]
            imagined_state = imagined_state.view(
                rollout_batch_size, *self.in_shape)
            imagined_state = self.target2pix(imagined_state)

            imagined_reward = F.softmax(imagined_reward, dim=1).max(dim=1)[1]

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
