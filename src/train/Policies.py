from random import randint, uniform
from typing import Tuple

import torch
import torch.nn.functional as F


class TrajCollectionPolicy:
    """
    Policy class called when collecting trajectories
    """

    def act(self, agent_id: str, observation: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
        """

        :param agent_id: the agent name as string
        :param observation: an observation related to the agent
        :return:
            action : an integer representing a discrete action
            value : the value associated with the action
            action_probs : a tensor of dim [batch size, action dim] having the softmax of the action logits

        """
        raise NotImplementedError


class RandomAction(TrajCollectionPolicy):

    def __init__(self, num_actions, device):
        self.num_actions = num_actions
        self.device = device

    def act(self, agent_id: str, observation: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
        action = randint(0, self.num_actions - 1)
        value = 0
        action_probs = torch.ones((1, self.num_actions))
        action_probs = action_probs.to(self.device)

        return action, value, action_probs


class ExplorationMAS(TrajCollectionPolicy):

    def __init__(self, ac_dict, num_actions):
        self.ac_dict = ac_dict
        self.num_actions = num_actions

        self.epsilon = 1
        self.decrease = 1e-4

    def act(self, agent_id: str, observation: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
        action_logit, value_logit = self.ac_dict[agent_id](observation)
        action_probs = F.softmax(action_logit, dim=1)

        if uniform(0, 1) < self.epsilon:
            action = randint(0, self.num_actions - 1)  # Explore action space
        else:
            action = action_probs.multinomial(1).squeeze()

        value = int(value_logit)
        action = int(action)

        self.epsilon -= self.decrease

        return action, value, action_probs


class MultimodalMAS(TrajCollectionPolicy):

    def __init__(self, ac_dict):
        self.ac_dict = ac_dict

    def act(self, agent_id: str, observation: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
        action_logit, value_logit = self.ac_dict[agent_id](observation)
        action_probs = F.softmax(action_logit, dim=1)
        action = action_probs.multinomial(1).squeeze()

        value = int(value_logit)
        action = int(action)

        return action, value, action_probs