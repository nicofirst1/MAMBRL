import itertools

import numpy as np
from ray.rllib.env import ParallelPettingZooEnv

from PettingZoo.pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv

from .Scenario import Scenario, is_collision
from .TimerLandmark import TimerLandmark


def get_env(kwargs) -> ParallelPettingZooEnv:
    """Initialize rawEnv and wrap it in parallel petting zoo"""
    env = RawEnv(**kwargs)
    env = ParallelPettingZooEnv(env)
    return env


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class RawEnv(SimpleEnv):
    def __init__(
            self,
            name,
            scenario_kwargs,
            max_cycles,
            continuous_actions,
            gray_scale=False,
    ):
        scenario = Scenario(**scenario_kwargs)
        world = scenario.make_world()
        super().__init__(
            scenario,
            world,
            max_cycles,
            continuous_actions,
            #color_entities=TimerLandmark,
        )
        self.metadata["name"] = name
        self.gray_scale = gray_scale
        self.agents_dict = {agent.name: agent for agent in world.agents}

    def get_reward_range(self):
        return self.scenario.get_reward_range()

    def reset(self):
        super(RawEnv, self).reset()

        return self.observe()

    def observe(self):
        observation = self.render(mode="rgb_array")

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

        return self.observe(), self.rewards, self.dones, self.infos
