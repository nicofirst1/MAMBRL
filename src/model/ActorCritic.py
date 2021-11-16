import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

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
        probs = F.softmax(logit)

        if deterministic:
            action = probs.max(1)[1]
        else:
            action = probs.multinomial(1)

        return action

    def evaluate_actions(self, x, action):
        logit, value = self.forward(x)

        probs = F.softmax(logit)
        log_probs = F.log_softmax(logit)

        action_log_probs = log_probs.gather(1, action)
        entropy = -(probs * log_probs).sum(1).mean()

        return logit, action_log_probs, value, entropy


class ActorCritic(OnPolicy):
    """
    This class is responsible for choosing an action and assigning a value given a state
    """
    def __init__(self, in_shape, num_actions):
        super(ActorCritic, self).__init__()

        self.in_shape = in_shape

        self.features = nn.Sequential(
            nn.Conv2d(in_shape[0], 16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 256),
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

    def feature_size(self):
        return self.features(autograd.Variable(torch.zeros(1, *self.in_shape))).view(1, -1).size(1)