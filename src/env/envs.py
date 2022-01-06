import itertools
from typing import Dict, Tuple

import numpy as np
import torch
from ray.rllib.utils.images import rgb2gray

from PettingZoo.pettingzoo.mpe._mpe_utils import rendering
from PettingZoo.pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from env.scenarios import CollectLandmarkScenario


class CollectLandmarkEnv(SimpleEnv):
    def __init__(self, scenario_kwargs: Dict, horizon, continuous_actions: bool,
                 gray_scale=False, frame_shape=None, visible=False
                 ):
        """
        This class has to manage the interaction between the agents in an environment.
        The env is made of N agents and M landmarks.
        The goal of the agents is to get to as many landmarks as possible in the shortest time
        Args:
            scenario_kwargs: dict of keyward for scenario initialization
            horizon: max steps before __all_done__=true
            continuous_actions: wether to use continous or discrete action
            gray_scale: if to convert obs to gray scale
            frame_shape: shape of a single frame
        """

        scenario = CollectLandmarkScenario(**scenario_kwargs)
        world = scenario.make_world()
        super().__init__(scenario, world, max_cycles=horizon, continuous_actions=continuous_actions,
                         local_ratio=None)  # color_entities=TimerLandmark

        self.frame_shape = frame_shape
        self.render_geoms = None
        self.gray_scale = gray_scale

        self.agent_selection = None
        self.agents_dict = {agent.name: agent for agent in world.agents}

        self.viewer = rendering.Viewer(frame_shape[1], frame_shape[2], visible=visible)
        self.viewer.set_max_size(scenario_kwargs['max_size'])

    def reset(self):
        super(CollectLandmarkEnv, self).reset()
        return self.observe()

    @property
    def action_meaning_dict(self):

        return {
            0: "stop",
            1: "left",
            2: "right",
            3: "up",
            4: "down"
        }

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
                observation = observation.permute(2, 0, 1)  ## fixme: a che serve questo?
            else:
                observation = observation.unsqueeze(dim=0)

            if self.gray_scale:
                observation = rgb2gray(observation)
                observation = np.expand_dims(observation, axis=0)

        return observation

    def step(self, actions: Dict[str, int]) -> Tuple[torch.Tensor, Dict[str, int], bool, Dict[str, Dict]]:
        """
        Takes a step in the environment.
        All the agents act simultaneously and the observation are collected
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
            super(CollectLandmarkEnv, self).step(action)

        # update landmarks status
        visited_landmarks = set(itertools.chain(self.scenario.visited_landmarks))
        self.scenario.visited_landmarks = []

        if len(visited_landmarks) > 0:
            self.render_geoms = None
            self.scenario.num_landmarks -= len(visited_landmarks)

            for lndmrk_id in visited_landmarks:
                landmark = self.scenario.landmarks[lndmrk_id]
                self.world.entities.remove(landmark)
                self.world.landmarks.remove(landmark)

        done = True if self.scenario.num_landmarks <= 0 else False
        observation = self.observe()

        return observation, self.rewards, done, {}


def get_env(kwargs: Dict) -> CollectLandmarkEnv:
    """Initialize rawEnv and wrap it in parallel petting zoo."""
    env = CollectLandmarkEnv(**kwargs)
    return env
