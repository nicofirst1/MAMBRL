import numpy as np
from ray.rllib.env import ParallelPettingZooEnv

from PettingZoo.pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from .Scenario import Scenario
from .TimerLandmark import TimerLandmark


def get_env(kwargs):
    """ Initialize rawEnv and wrap it in parallel petting zoo"""
    env = RawEnv(**kwargs)
    env = ParallelPettingZooEnv(env)
    return env


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


class RawEnv(SimpleEnv):
    def __init__(
            self, name, N, landmarks, max_cycles, continuous_actions, gray_scale=False
    ):
        scenario = Scenario()
        world = scenario.make_world(N, landmarks)
        super().__init__(
            scenario,
            world,
            max_cycles,
            continuous_actions,
            color_entities=TimerLandmark,
        )
        self.metadata["name"] = name
        self.gray_scale = gray_scale
        self.agents_dict={agent.name:agent for agent in world.agents}

    @staticmethod
    def is_collision(agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

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

            for landmark in self.world.landmarks:

                visited = False

                # check if one agent has visited the landmark
                if self.is_collision(self.agents_dict[agent_id], landmark):
                    visited = True
                    break

                # if visited reset else increase
                if visited:
                    landmark.reset_timer()
                else:
                    landmark.step()

        return self.observe(), self.rewards, self.dones, self.infos
