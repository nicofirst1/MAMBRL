import math

import torch
import torch.nn as nn
from torchvision.transforms import transforms

from common.distributions import Categorical
from src.common import FixedCategorical, init

class Flatten(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}

        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape, **base_kwargs)
        self.dist = Categorical(self.base.output_size, action_space)

    def get_modules(self):
        return self.base.get_modules()

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        if actor_features.isnan().any():
            print()

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        return self.base(inputs, rnn_hxs, masks)[0]

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs

    def get_modules(self):
        return {}


class CNNBase(NNBase):
    def __init__(self, input_shape, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x:
            nn.init.constant_(x, 0), nn.init.calculate_gain("relu"))

        num_inputs = input_shape[0]
        middle_shape = input_shape[1:]
        middle_shape = (middle_shape[0] // 4 - 1, middle_shape[1] // 4 - 1)
        middle_shape = (middle_shape[0] // 2 - 1, middle_shape[1] // 2 - 1)
        middle_shape = (middle_shape[0] - 2, middle_shape[1] - 2)

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * middle_shape[0] * middle_shape[1], hidden_size)), nn.ReLU())

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critic_linear = init_(nn.Linear(hidden_size, 1))
        self.train()

    def get_modules(self):
        return dict(
            value=self.critic_linear,
            features=self.main
        )

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class ResNetBase(NNBase):
    def __init__(self, input_shape, recurrent=False, hidden_size=512):
        super(ResNetBase, self).__init__(recurrent, hidden_size, hidden_size)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406] * 4, std=[0.229, 0.224, 0.225] * 4
                ),
            ]
        )
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # remove last linear layer
        self.features = (list(model.children())[:-1])

        # add initial convolution for stacked frames input
        num_inputs = input_shape[0]

        self.conv_0 = init_(
            nn.Conv2d(num_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        )
        self.features[0] = self.conv_0

        end = self.features[-1]
        # get up to the N+1 layer and discard the rest
        self.features = self.features[:7]
        # add average pool as last
        self.features[-1] = end
        self.features = torch.nn.Sequential(*self.features)

        self.hidden_layer = nn.Sequential(nn.Linear(128, hidden_size),
                                          nn.ReLU())

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1))

        self.train()

    def get_modules(self):
        return dict(
            value=self.classifier,
            hidden_layer=self.hidden_layer,
            features=self.features
        )

    def forward(self, inputs, rnn_hxs, masks):
        x = self.preprocess(inputs / 255.0)
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.hidden_layer(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.classifier(x), x, rnn_hxs
