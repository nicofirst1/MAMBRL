import torch
import torch.nn as nn

from src.model.ModelFree import OnPolicy
from src.model.RolloutEncoder import RolloutEncoder


class I2A(OnPolicy):
    def __init__(self, in_shape, num_actions, num_rewards, hidden_size, imagination, num_frames, full_rollout=True):
        super(I2A, self).__init__()

        self.in_shape = in_shape
        self.num_actions = num_actions
        self.num_rewards = num_rewards

        self.imagination = imagination
        num_channels = in_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(num_channels * num_frames, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.encoder = RolloutEncoder(in_shape, num_rewards, hidden_size,num_frames=num_frames)

        features_out = self.features(torch.zeros(1, num_channels * num_frames, *self.in_shape[1:])).view(1, -1).size(1)

        if full_rollout:
            self.fc = nn.Sequential(
                nn.Linear(features_out + num_actions * hidden_size, 256),
                nn.ReLU(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(features_out + hidden_size, 256),
                nn.ReLU(),
            )

        self.critic = nn.Linear(256, 1)
        self.actor = nn.Linear(256, num_actions)

    def forward(self, state):
        batch_size = state.size(0)
        imagined_state, imagined_reward = self.imagination(state)
        hidden = self.encoder(imagined_state, imagined_reward)
        hidden = hidden.view(batch_size, -1)

        state = self.features(state)
        state = state.view(state.size(0), -1)

        x = torch.cat([state, hidden], 1)
        x = self.fc(x)

        action_logit = self.actor(x)
        value_logit = self.critic(x)

        return action_logit, value_logit
