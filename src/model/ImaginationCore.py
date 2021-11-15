import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    Here this has to be clarified.. what is this pixels and how the function
    target_to_pix is used!
"""
pixels = (
    (0.0, 1.0, 0.0),
    (0.0, 1.0, 1.0),
    (0.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, 0.0),
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0)
)
pixel_to_onehot = {pix:i for i, pix in enumerate(pixels)}

def target_to_pix(imagined_states):
    pixels = []
    """
        To pixel will become this:
            {0: (0.0, 1.0, 0.0), 1: (0.0, 1.0, 1.0), 2: (0.0, 0.0, 1.0), 3: (1.0, 1.0, 1.0), 4: (1.0, 1.0, 0.0), 5: (0.0, 0.0, 0.0), 6: (1.0, 0.0, 0.0)}
    
        so this means that the imagined_states coming as input should have values only
        in the range 0 - 6. This is weird!
        N.B. with paras.out_shape = (3, 32, 32), the input imagined_states has shape (1024,)
    """
    to_pixel = {value: key for key, value in pixel_to_onehot.items()}
    for target in imagined_states:
        pixels.append(list(to_pixel[target]))
    return np.array(pixels)


class ImaginationCore(nn.Module):
    def __init__(self, num_rolouts, in_shape, num_actions, num_rewards, env_model, distil_policy, device, full_rollout=True):
        super().__init__()
        self.num_rolouts = num_rolouts
        self.in_shape = in_shape
        self.num_actions = num_actions
        self.num_rewards = num_rewards
        self.env_model = env_model
        self.distil_policy = distil_policy
        self.full_rollout = full_rollout
        self.device= device

    def forward(self, state):
        # state      = state.cpu()
        batch_size = state.size(0)

        rollout_states = []
        rollout_rewards = []

        if self.full_rollout:
            state = state.unsqueeze(0).repeat(self.num_actions, 1, 1, 1, 1).view(-1, *self.in_shape)
            action = torch.LongTensor([[i] for i in range(self.num_actions)] * batch_size)
            action = action.view(-1)
            rollout_batch_size = batch_size * self.num_actions
        else:
            # get action based on current state
            action = self.distil_policy.act(state)
            action = action.detach()
            rollout_batch_size = batch_size

        for step in range(self.num_rolouts):
            # propagate action on "image" (aka tensor of same shape as image)
            # this tensor
            onehot_action = torch.zeros(rollout_batch_size, self.num_actions, *self.in_shape[1:]).to(self.device)
            onehot_action[range(rollout_batch_size), action] = 1
            inputs = torch.cat([state, onehot_action], 1).to(self.device)

            imagined_state, imagined_reward = self.env_model(inputs)

            imagined_state = F.softmax(imagined_state, dim=1).max(1)[1]
            imagined_reward = F.softmax(imagined_reward, dim=1).max(1)[1]

            """
                Commentend cause it does not work!!
            """
            #imagined_state = target_to_pix(imagined_state.detach().cpu().numpy())
            #imagined_state = torch.FloatTensor(imagined_state).view(rollout_batch_size, *self.in_shape).to(self.device)

            """
                Added by me to go back from tensor of (1024,) to image of (3, 32, 32)
                
                Qui ci sta un problema -> dentro il rollout viene inserito lo stato
                con dimensione (1, 1, 3, 32, 32) (riga 102) e questo poi da problemi quando vengono
                settati più num_step (problemi sulla chiamata riga 111 di train.py),
                perché nel rollout se ci sono due stati lo shape sarà (2, 1, 3, 32, 32)
                e questo shape non piace alla rete neurale. Bisogna capire dove vanno
                aggiustate le cose, perché la chiamata riga 44 di I2A.py (chiamata all'env_model)
                sembra volere lo shape (1, 1, 3, 32, 32) quindi quest'ultimo sembra essere giusto!
            """
            imagined_state = imagined_state.detach().cpu().numpy()
            imagined_state = np.concatenate([imagined_state, imagined_state, imagined_state])
            imagined_state = torch.FloatTensor(np.reshape(imagined_state, (-1, 3, 32, 32)))

            onehot_reward = torch.zeros(rollout_batch_size, self.num_rewards)
            onehot_reward[range(rollout_batch_size), imagined_reward.detach().cpu().numpy()] = 1

            rollout_states.append(imagined_state.unsqueeze(0))
            #rollout_states.append(imagined_state)
            rollout_rewards.append(onehot_reward.unsqueeze(0))

            state = imagined_state.to(self.device)
            action = self.distil_policy.act(state)
            action = action.detach()

        return torch.cat(rollout_states).to(self.device), torch.cat(rollout_rewards).to(self.device)
