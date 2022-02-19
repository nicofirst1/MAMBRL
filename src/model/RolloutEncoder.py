import torch
import torch.nn as nn
from collections import OrderedDict


class RolloutEncoder(nn.Module):
    def __init__(self, in_shape, len_rewards, conv_layers, hidden_size=256, **kwargs):
        super(RolloutEncoder, self).__init__()
        self.in_shape = in_shape

        next_inp = None
        feature_extractor_layers = OrderedDict()
        for i, cnn in enumerate(conv_layers):
            if i == 0:
                feature_extractor_layers["re_conv_0"] = nn.Conv2d(
                    self.in_shape[0], cnn[0], kernel_size=cnn[1], stride=cnn[2])
                feature_extractor_layers["re_conv_0_activ"] = nn.ReLU()
            else:
                feature_extractor_layers["re_conv_" + str(i)] = nn.Conv2d(
                    next_inp, cnn[0], kernel_size=cnn[1], stride=cnn[2])
                feature_extractor_layers["re_conv_" +
                                         str(i) + "_activ"] = nn.ReLU()
            next_inp = cnn[0]
        # compute the input shape for the residual network
        for layer in feature_extractor_layers:
            if layer == "re_conv_0":
                fake_inp = torch.zeros(
                    [1, *self.in_shape])
                fake_inp = feature_extractor_layers[layer](fake_inp)
            else:
                fake_inp = feature_extractor_layers[layer](fake_inp)

        # flatten layer
        next_inp = fake_inp.view(1, -1).size(1)
        feature_extractor_layers["re_flatten"] = nn.Flatten()
        self.feature_extractor = nn.Sequential(feature_extractor_layers)
        self.re_LSTM = nn.LSTM(
            next_inp + len_rewards, hidden_size)

    def forward(self, state, reward):
        """forward method.

        This network should receive as input a series of predicted future
        frames and reward concatenated along the 0th axis
        Parameters
        ----------
        state : Tensor
            tensor of shape [sequence_len, batch_size, num_channels, width, height]
        reward : Tensor
            tensor of shape [sequence_len, batch_size, len_reward]

        Returns
        -------
        Tensor
            Returns a tensor of size: [batch_size, num_features] where
            num_features is the hidden size of the last recurrent layer.

        """
        num_steps = state.size(0)
        batch_size = state.size(1)
        state = state.view(-1, *self.in_shape)
        state = self.feature_extractor(state)
        state = state.view(num_steps, batch_size, -1)
        # state has 3136 feature and reward is just 1, maybe the reward is
        # useless in our setting
        lstm_input = torch.cat([state, reward], dim=2)
        out, hidden = self.re_LSTM(lstm_input)

        # hidden is a tuple (h_n, c_n) where h_n is the final hidden state of
        # all the layers and c_n is the final cell state for all the layers
        return hidden[0].squeeze(0)


if __name__ == "__main__":
    from src.common.Params import Params
    params = Params()
    re_params = params.get_rollout_encoder_configs()
    re_model = RolloutEncoder(**re_params)
    # The input to the rollout encoder should be a tensor of size:
    # [sequence_len, batch_size, channel_size, width, height]
    re_state_input = torch.rand(params.future_frame_horizon,
                                params.batch_size, *params.frame_shape)
    re_reward_input = torch.rand(params.future_frame_horizon,
                                 params.batch_size, params.len_reward)

    out = re_model(re_state_input, re_reward_input)
