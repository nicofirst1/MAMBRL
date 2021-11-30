import itertools
from copy import copy

import numpy as np
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
        mode="human",
    ):
        scenario = Scenario(**scenario_kwargs)
        world = scenario.make_world()
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

        visible = True if mode == "human" else False
        self.viewer = rendering.Viewer(700, 700, visible=visible)
        self.viewer.set_max_size(3)

    def get_reward_range(self):
        return self.scenario.get_reward_range()

    def reset(self, mode="human"):
        super(RawEnv, self).reset()

        return self.observe(mode=mode)

    def observe(self, mode):

        observation = self.render(mode=mode)

        if self.gray_scale and observation is not None:
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

        observation = self.observe(mode="rgb_array")
        # copy done so __all__ is not appended
        dones = copy(self.dones)
        dones["__all__"] = all(dones.values())

        # add agent state to infos
        self.infos = {k: self.agents_dict[k].state for k in self.infos.keys()}

        return observation, self.rewards, dones, self.infos


def get_env(kwargs) -> RawEnv:
    """Initialize rawEnv and wrap it in parallel petting zoo"""
    env = RawEnv(**kwargs)
    return env
