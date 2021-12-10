import itertools
from copy import copy

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
            name,
            scenario_kwargs,
            max_cycles,
            continuous_actions,
            gray_scale=False,
            obs_shape=None,
            mode="human",
    ):
        scenario = Scenario(**scenario_kwargs)
        max_size = 3
        world = scenario.make_world(max_size)
        super().__init__(
            scenario,
            world,
            max_cycles,
            continuous_actions,
            # color_entities=TimerLandmark,
        )
        self.metadata["name"] = name
        self.gray_scale = gray_scale
        self.agents_dict = {agent.name: agent for agent in world.agents}
        self.obs_shape = obs_shape

        visible = True if mode == "human" else False
        self.mode=mode
        self.viewer = rendering.Viewer(obs_shape, obs_shape, visible=visible)
        self.viewer.set_max_size(max_size)

    def get_reward_range(self):
        return self.scenario.get_reward_range()

    def reset(self):
        super(RawEnv, self).reset()

        return self.observe()

    def observe(self):

        observation = self.render(mode=self.mode)

        if observation is not None:

            observation = torch.from_numpy(observation.copy())
            observation= observation.float()
            # move channel on second dimension if present, else add 1
            if len(observation.shape) == 3:
                observation = observation.permute(2, 0, 1)
            else:
                observation = observation.unsqueeze(dim=0)

            if self.gray_scale:
                observation = rgb2gray(observation)
                observation = np.expand_dims(observation, axis=0)

        return observation

    def step(self, actions):

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


def get_env(kwargs) -> RawEnv:
    """Initialize rawEnv and wrap it in parallel petting zoo."""
    env = RawEnv(**kwargs)
    return env
