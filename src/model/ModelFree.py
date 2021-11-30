import torch
import torch.nn as nn
import torch.nn.functional as F


class OnPolicy(nn.Module):
    def __init__(self):
        super(OnPolicy, self).__init__()

    def forward(self, x):
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

    # fix: check if it works
    def evaluate_actions(self, frames, action):
        """evaluate_actions method.

        compute the actions logit, value and actions probability based on the
        the actual state and then compute the entropy with respect to the 
        action that we passed as a parameter. 
        Parameters
        ---------
        """
        action_logit, value = self.forward(frames)

        probs = F.softmax(action_logit, dim=1)

        log_probs = F.log_softmax(action_logit, dim=1)

        action_log_probs = log_probs.gather(1, action)
        entropy = -(probs * log_probs).sum(1).mean()

        return action_logit, probs, value, entropy


class ModelFree(OnPolicy):
    """ModelFree class.

    This class is responsible for choosing an action and assigning a value
    given a state
    """

    def __init__(self, in_shape, num_actions):
        super(ModelFree, self).__init__()

        self.in_shape = in_shape
        num_channels = in_shape[0]

        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        features_out = (
            self.features(torch.zeros(1, num_channels, *self.in_shape[1:]))
            .view(1, -1)
            .size(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(features_out, 256),
            nn.ReLU(),
        )

        self.critic = nn.Linear(256, 1)
        self.actor = nn.Linear(256, num_actions)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        logit = self.actor(x)
        value = self.critic(x)
        return logit, value
