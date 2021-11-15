import torch
import torch.nn as nn

from src.model.ActorCritic import OnPolicy
from src.model.RolloutEncoder import RolloutEncoder


class I2A(OnPolicy):
    def __init__(self, in_shape, num_actions, num_rewards, hidden_size, imagination, full_rollout=True):
        super(I2A, self).__init__()

        self.in_shape = in_shape
        self.num_actions = num_actions
        self.num_rewards = num_rewards

        self.imagination = imagination

        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.encoder = RolloutEncoder(in_shape, num_rewards, hidden_size)

        if full_rollout:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + num_actions * hidden_size, 256),
                nn.ReLU(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.feature_size() + hidden_size, 256),
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

        logit = self.actor(x)
        value = self.critic(x)

        return logit, value

    def feature_size(self):
        return self.features(torch.zeros(1, *self.in_shape)).view(1, -1).size(1)