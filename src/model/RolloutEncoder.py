import torch
import torch.nn as nn


class RolloutEncoder(nn.Module):
    def __init__(self, in_shape, num_rewards, hidden_size, num_frames):
        super(RolloutEncoder, self).__init__()
        self.in_shape = list(in_shape)


        self.features = nn.Sequential(
            nn.Conv2d(self.in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        features_out = self.features(torch.zeros(1, *self.in_shape)).view(1, -1).size(1)

        self.gru = nn.GRU(features_out + num_rewards, hidden_size)

    def forward(self, state, reward):
        num_steps = state.size(0)
        batch_size = state.size(1)

        state = state.view(-1, *self.in_shape)
        state = self.features(state)
        state = state.view(num_steps, batch_size, -1)
        rnn_input = torch.cat([state, reward], 2)
        _, hidden = self.gru(rnn_input)
        return hidden.squeeze(0)

    def feature_size(self):
        return self.features(torch.zeros(1, *self.in_shape)).view(1, -1).size(1)
