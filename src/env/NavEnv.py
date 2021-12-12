import itertools
from copy import copy
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ray.rllib.utils.images import rgb2gray

from PettingZoo.pettingzoo.mpe._mpe_utils import rendering
from PettingZoo.pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from .Scenario import Scenario


class RawEnv(SimpleEnv):
    def __init__(
            self,
            name: str,
            scenario_kwargs: Dict,
            horizon,
            continuous_actions: bool,
            gray_scale=False,
            obs_shape=None,
            visible=False,
    ):
        """
        This class has to manage the interaction between the agents in an environment.
        The env is made of N agents and M landmarks.
        The goal of the agents is to get to as many landmarks as possible in the shortest time
        Args:
            name: name of the env
            scenario_kwargs: dict of keyward for scenario initialization
            horizon: max steps before __all_done__=true
            continuous_actions: if to use continous or discrete actions
            gray_scale: if to convert obs to gray scale
            obs_shape: shape of observation space, used for rescaling
            mode: rendering mode, either human or rgb_array
        """
        scenario = Scenario(**scenario_kwargs)

        world = scenario.make_world()
        super().__init__(
            scenario,
            world,
            max_cycles=horizon,
            continuous_actions=continuous_actions,
            # color_entities=TimerLandmark,
        )
        self.metadata["name"] = name
        self.gray_scale = gray_scale
        self.agents_dict = {agent.name: agent for agent in world.agents}
        self.obs_shape = obs_shape


        self.viewer = rendering.Viewer(obs_shape, obs_shape, visible=visible)
        self.viewer.set_max_size(scenario_kwargs['max_size'])

    def get_reward_range(self) -> List[int]:
        return self.scenario.get_reward_range()

    def reset(self):
        super(RawEnv, self).reset()

        self.scenario.reset_world(self.world, self.np_random)

        for lndmrk_id in self.scenario.landmarks.values():
            lndmrk_id.reset_counter()

        return self.observe()

    def observe(self, agent="") -> torch.Tensor:
        """
        Get an image of the game
        Args:
            agent: All observation are the same here

        Returns: returns an image as a torch tensor of size [channels, width, height]

        """

        observation = self.render()

        if observation is not None:

            observation = torch.from_numpy(observation.copy())
            observation = observation.float()
            # move channel on second dimension if present, else add 1
            if len(observation.shape) == 3:
                observation = observation.permute(2, 0, 1)
            else:
                observation = observation.unsqueeze(dim=0)

            if self.gray_scale:
                observation = rgb2gray(observation)
                observation = np.expand_dims(observation, axis=0)

        return observation

    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, torch.Tensor], Dict[str, int], Dict[str, bool], Dict[
        str, Dict]]:
        """
        Takes a step in the environment.
        All the agents act simultaneously and the the observation are collected
        Args:
            actions: dictionary mapping angent name to an action

        Returns: all returns are dict mapping agent string to a value
            observation : the observed window as a torch.Tensor
            rewards: a reward as an int
            dones: if the agent is done or not
            infos: additional infos on the agent, such as its position

        """

        for agent_id, action in actions.items():
            self.agent_selection = agent_id
            super(RawEnv, self).step(action)

        # update landmarks status
        visited_landmarks = list(self.scenario.registered_collisions.values())
        visited_landmarks = set(itertools.chain(*visited_landmarks))

        not_visited = set(self.scenario.landmarks.keys()) - visited_landmarks

        for lndmrk_id in visited_landmarks:
            self.scenario.landmarks[lndmrk_id].reset_counter()
        for lndmrk_id in not_visited:
            self.scenario.landmarks[lndmrk_id].step()

        observation = self.observe()
        # copy done so __all__ is not appended
        dones = copy(self.dones)
        dones["__all__"] = all(dones.values())

        # add agent state to infos
        self.infos = {k: self.agents_dict[k].state for k in self.infos.keys()}

        return observation, self.rewards, dones, self.infos


def get_env(kwargs:Dict
            ) -> RawEnv:
    """Initialize rawEnv and wrap it in parallel petting zoo."""
    env = RawEnv(**kwargs)
    return env
