from random import randint, uniform
from typing import Tuple

import torch
import torch.nn.functional as F


class TrajCollectionPolicy:
    """
    Policy class called when collecting trajectories
    """

    def act(
        self, agent_id: str, observation: torch.Tensor
    ) -> Tuple[int, int, torch.Tensor]:
        """

        :param agent_id: the agent name as string
        :param observation: an observation related to the agent
        :return:
            action : an integer representing a discrete action
            value : the value associated with the action
            action_log_probs : a tensor of dim [batch size, action dim] having the log_softmax of the action logits

        """
        raise NotImplementedError


class RandomAction(TrajCollectionPolicy):
    def __init__(self, num_actions, device):
        self.num_actions = num_actions
        self.device = device

    def act(
        self, agent_id: str, observation: torch.Tensor
    ) -> Tuple[int, int, torch.Tensor]:
        action = randint(0, self.num_actions - 1)
        value = 0
        action_probs = torch.ones((1, self.num_actions))
        action_probs = action_probs.to(self.device)

        return action, value, action_probs


class EpsilonGreedy(TrajCollectionPolicy):
    def __init__(self, ac_dict, num_actions):
        self.ac_dict = ac_dict
        self.num_actions = num_actions

        self.epsilon = 1
        self.decrease = 5e-5

    def act(
        self, agent_id: str, observation: torch.Tensor
    ) -> Tuple[int, float, torch.Tensor, bool]:
        """act method.

               implementation of the epsilon greedy exploration function
               Returns
               -------
               action: int
                   index of the action chosen
               value: float
                   value of the state
               log_actions_prob: torch.Tensor
                   [self.num_actions] log actions prob
               rand_act: bool
                   if the action was chosen randomly
               """
        # action_logit [1, num_action] value_logit [1,1]
        action_logit, value_logit = self.ac_dict[agent_id](observation)
        action_probs = F.softmax(action_logit, dim=1)

        if uniform(0, 1) < self.epsilon:
            action = randint(0, self.num_actions - 1)  # Explore action space
            rand_act = True

        else:
            action = action_probs.max(1)[1]
            rand_act = False

        value = int(value_logit)
        action = int(action)

        # fixme: check if are identical
        log_action_prob = torch.log(action_probs).mean()
        log_action_prob = F.log_softmax(action_logit, dim=1).squeeze()

        if self.epsilon > 0:
            self.epsilon -= self.decrease

        return action, value, log_action_prob, rand_act

    def increase_temp(self, actions: torch.Tensor):
        var = actions.float().var()
        # +1 because actions id starts from 0
        mean = (actions + 1).float().mean()

        if not mean - var < self.num_actions / 2 < mean + var:
            if self.epsilon < 1:
                self.epsilon += 0.2


class MultimodalMAS(TrajCollectionPolicy):

    def __init__(self, ac_dict, cr_dict=None, share_weights=True):
        self.share_weights = share_weights
        if self.share_weights:
            self.ac_dict = ac_dict
        else:
            assert cr_dict is not None, \
                f"{cr_dict} is invalid. You need to specify a valid model for the critic"
            self.ac_dict = ac_dict
            self.cr_dict = cr_dict

    def act(self, agent_id: str, observation: torch.Tensor) -> Tuple[int, float, torch.Tensor]:

        if self.share_weights:
            action_logit, value_logit = self.ac_dict[agent_id](observation)
        else:
            action_logit = self.ac_dict[agent_id](observation)
            value_logit = self.cr_dict[agent_id](observation)

        action_probs = F.softmax(action_logit, dim=1)
        action = action_probs.multinomial(1)

        log_actions_prob = F.log_softmax(action_logit, dim=1).squeeze()

        value = float(value_logit)
        action = int(action)

        return action, value, log_actions_prob
