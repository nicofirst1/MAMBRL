import torch
import torch.nn as nn

from torch.nn import Flatten
from common.distributions import Categorical, FixedCategorical

from model.utils import init

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

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False, full_log_prob=False):
        value, actor_features = self.base(inputs)
        logits = self.dist(actor_features)
        dist = FixedCategorical(logits=logits)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        if full_log_prob:
            action_log_probs = torch.log_softmax(logits, dim=-1)
        else:
            action_log_probs = dist.log_probs(action)

        # dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        return self.base(inputs)[0]

    def evaluate_actions(self, inputs, action, full_log_prob=False):
        value, actor_features = self.base(inputs)
        logits = self.dist(actor_features)
        dist = FixedCategorical(logits=logits)

        if full_log_prob:
            action_log_probs = torch.log_softmax(logits, dim=-1)
        else:
            action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class NNBase(nn.Module):
    def __init__(self, hidden_size):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        return 1

    @property
    def output_size(self):
        return self._hidden_size

class CNNBase(NNBase):
    def __init__(self, input_shape, hidden_size=512):
        super(CNNBase, self).__init__(hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

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

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs):
        x = self.main(inputs / 255.0)
        return self.critic_linear(x), x