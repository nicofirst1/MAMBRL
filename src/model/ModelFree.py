from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms


class OnPolicy(nn.Module):
    def __init__(self, num_actions, features_out=256):
        super(OnPolicy, self).__init__()

        self._features_out=features_out
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

    def act(self, x, deterministic=True):
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

    def evaluate_actions(self, frames: torch.Tensor, action_indx: torch.Tensor):
        """evaluate_actions method.

        compute the actions logit, value and actions probability by passing
        the actual states (frames) to the ModelFree network. Then it computes
        the entropy of the action corresponding to the index action_indx

        Parameters
        ----------
        frames : PyTorch Array
            a 4 dimensional tensor [batch_size, channels, width, height]
        action_indx : torch.Tensor
            Tensor [batch_size] index of the actions to use in order to compute the entropy

        Returns
        -------
        action_logit : Torch.Tensor [batch_size, num_actions]
            output of the ModelFree network before passing it to the softmax
        action_log_prob : Torch.Tensor [batch_size,1]
            scalar value, action log of the action corresponding to the
            action_indx
        probs : Torch.Tensor [batch_size, num_actions]
            probability of actions given by the ModelFree network
        value : Torch.Tensor [batch_size,1]
            value of the state given by the ModelFree network
        entropy : Torch.Tensor [batch_size,1]
            value of the entropy given by the action with index equal to
            action_indx.
        """
        action_logit, value = self.forward(frames)

        action_probs = F.softmax(action_logit, dim=1)

        log_probs = F.log_softmax(action_logit, dim=1)

        if action_indx.ndim == 1:
            action_indx = action_indx.unsqueeze(1)

        action_log_prob = log_probs.gather(1, action_indx)
        entropy = -(action_probs * log_probs).sum(1).mean()

        return action_logit, action_log_prob, action_probs, value, entropy


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
        x = x.view(x.size(0), -1)
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
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        self.features = model

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.preprocess(x)
        return super(ModelFreeResnet, self).forward(x)
