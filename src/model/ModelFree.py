from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Flatten

from common.distributions import Categorical, FixedCategorical
from torchvision.transforms import transforms

from common.utils import init


class OnPolicy(nn.Module):
    def __init__(self, num_actions, features_out=256):
        super(OnPolicy, self).__init__()

        self._features_out = features_out
        self.critic = nn.Linear(features_out, 1)
        self.actor = nn.Linear(features_out, num_actions)

    def forward(self, input) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass extracting the actor features and returning the values
        :param input:
        :return:
            action_logits: a torch tensor of size [batch_size, num_actions]
            values: the value from the value layer

        """
        raise NotImplementedError

    def act(self, x, deterministic=False):
        """
        Use the current forward method to extract action based logits
        Then return an action in a deterministic or stochastic way
        """

        logit, value = self.forward(x)
        probs = F.softmax(logit, dim=0)

        if deterministic:
            action = probs.max(1)[1]
        else:
            action = probs.multinomial(1)

        return action

    def evaluate_actions(self, frames: torch.Tensor, actions_indxs: torch.Tensor):
        """evaluate_actions method.

        compute the actions logit, value and actions probability by passing
        the actual states (frames) to the ModelFree network. Then it computes
        the entropy of the action corresponding to the index action_indx

        Parameters
        ----------
        frames : PyTorch Array
            a 4 dimensional tensor [batch_size, channels, width, height]
        actions_indxs : torch.Tensor[batch_size] index of the actions to use in order to compute the entropy

        Returns
        -------
        action_logit : Torch.Tensor [batch_size, num_actions]
            output of the ModelFree network before passing it to the softmax
        action_log_probs : torch.Tensor [batch_size, num_actions]
            log_probs of all the actions
        probs : Torch.Tensor [batch_size, num_actions]
            probability of actions given by the ModelFree network
        value_logit : Torch.Tensor [batch_size,1]
            value of the state given by the ModelFree network
        entropy : Torch.Tensor [batch_size,1]
            value of the entropy given by the action with index equal to
            action_indx.
        """
        action_logit, value_logit = self.forward(frames)

        action_probs = F.softmax(action_logit, dim=1)
        action_probs_log = F.log_softmax(action_logit, dim=1)

        entropy = -(action_probs * action_probs_log).sum(1).mean()

        return action_logit, action_probs_log.gather(1, actions_indxs.unsqueeze(dim=1)), action_probs, value_logit, entropy


class ModelFree(OnPolicy):
    """ModelFree class.

    This class is responsible for choosing an action and assigning a value
    given a state
    """

    def __init__(self, in_shape, num_actions):
        super(ModelFree, self).__init__(num_actions)

        self.in_shape = in_shape
        num_channels = in_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        fc_in = (
            self.features(torch.zeros(1, num_channels, *self.in_shape[1:]))
            .view(1, -1)
            .size(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(fc_in, self._features_out),
            nn.ReLU(),
        )

        self.critic = nn.Linear(256, 1)
        self.actor = nn.Linear(256, num_actions)

    def to(self, device):
        self.features = self.features.to(device)
        self.fc = self.fc.to(device)
        self.critic = self.critic.to(device)
        self.actor = self.actor.to(device)

        return self

    def forward(self, input):
        """forward method.

        Return the logit and the values of the ModelFree
        Parameters
        ----------
        input : torch.Tensor
            [batch_size, num_channels, width, height]

        Returns
        -------
        logit : torch.Tensor
            [batch_size, num_actions]
        value : torch.Tensor
            [batch_size, value]

        """
        x = self.features(input)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        action_logits = self.actor(x)
        value = self.critic(x)
        return action_logits, value


class ModelFreeResnet(ModelFree):
    def __init__(self, **kwargs):
        super(ModelFreeResnet, self).__init__(**kwargs)

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model = torch.nn.Sequential(*(list(model.children())[:-1]))


        self.features = model

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )

    def train(self, mode: bool = True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            if module == self.features:
                module.eval()
            module.train(mode)
        return self

    def forward(self, x):
        x = self.preprocess(x)
        return super(ModelFreeResnet, self).forward(x)


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

        dist_entropy = dist.entropy().mean()

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