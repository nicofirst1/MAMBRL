from typing import Tuple

import numpy as np

from PettingZoo.pettingzoo.mpe._mpe_utils.core import Agent, Entity, World
from PettingZoo.pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from src.common import min_max_norm
from .timer_landmark import TimerLandmark


def is_collision(entity1, entity2):
    delta_pos = entity1.state.p_pos - entity2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = entity1.size + entity2.size
    return True if dist < dist_min else False


def get_distance(entity1, entity2):
    delta_pos = entity1.state.p_pos - entity2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))

    return dist


class Border(Entity):
    def __init__(
            self, start: Tuple[int, int], end: Tuple[int, int], color=(1, 0, 0), linewidth=1
    ):
        super(Border, self).__init__()
        self.start = np.array(start)
        self.end = np.array(end)
        self.color = np.array(color)
        self.linewidth = linewidth

        p_pos = (
            (start[0] + end[0]) / 2,
            (start[1] + end[1]) / 2,
        )

        self.state.p_pos = np.array(p_pos)


class BoundedWorld(World):
    """
    A bounded word for the env. Uses unmovable landmarks on the border of the image in order to bound the agents
    """

    def __init__(self, max_size: int):
        super(BoundedWorld, self).__init__()

        self.max_size = max_size
        max_size -= 0.2

        b1 = Border((-max_size, -max_size), (max_size, -max_size))
        b2 = Border((-max_size, -max_size), (-max_size, max_size))
        b3 = Border((max_size, -max_size), (max_size, max_size))
        b4 = Border((max_size, max_size), (-max_size, max_size))

        self.borders = [b1, b2, b3, b4]
        self.contact_margin = 0.1

    @property
    def entities(self):
        return self.landmarks + self.borders + self.agents


class CollectLandmarkScenario(BaseScenario):
    def __init__(
            self,
            num_agents: int,
            num_landmarks: int,
            max_size: int,
            step_reward: int,
            landmark_reward: int,
            np_random,
    ):
        """

        Args:
            num_agents: the number of agents in the system
            num_landmarks: the number of landmarks
            max_size: maximum size of the word, used in word initialization
        """

        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.max_size = max_size

        self.step_reward = step_reward
        self.landmark_reward = landmark_reward
        self.np_random = np_random

        self.landmarks = {}
        self.visited_landmarks = []
        (
            self.reward_curriculum,
            self.landmark_curriculum,
        ) = self.init_curriculum_learning()

    @staticmethod
    def init_curriculum_learning():

        reward_modalities = {
            0: "Reward is the (world.maxsize - distance between agent and closest landmark), +landmark_reward when agent on landmark",
            1: "Reward is 0 at every time step and +landmark_reward when agent on landmark",
            2: "Reward is -step_reward at every time step and +landmark_reward when agent on landmark",
            "current": 0,
        }

        landmark_modalities = {
            0: "Landmark have static dimension and positions",
            1: "Landmark have static dimension and random positions",
            2: "Landmark have random dimension and position",
            "current": 0,
        }

        return reward_modalities, landmark_modalities

    def set_curriculum(self, reward: int = None, landmark: int = None):
        if reward is not None:
            assert (
                    reward in self.reward_curriculum.keys()
            ), f"Reward curriculum modality '{reward}' is not in range"
            self.reward_curriculum["current"] = reward

        if landmark is not None:
            assert (
                    landmark in self.landmark_curriculum.keys()
            ), f"Landmark curriculum modality '{landmark}' is not in range"
            self.landmark_curriculum["current"] = landmark

    def get_curriculum(self) -> Tuple[Tuple[int, str], Tuple[int, str]]:
        r = self.reward_curriculum["current"]
        l = self.landmark_curriculum["current"]

        return (r, self.reward_curriculum[r]), (l, self.landmark_curriculum[l])

    def make_world(self) -> World:
        """
        Init world and populate it with agents and landmarks
        Returns:

        """
        world = BoundedWorld(self.max_size)

        # set any world properties first
        world.dim_c = 2

        # add agents
        world.agents = [Agent() for _ in range(self.num_agents)]

        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.1
            agent.accel = 4.0
            agent.max_speed = 1.3
            agent.color = np.array([0, 0, 1])

        # add agents collisions
        self.visited_landmarks = []

        # add landmarks
        world.landmarks = [
            TimerLandmark(self.np_random) for _ in range(self.num_landmarks)
        ]
        landmark_pos = {}
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark_{i}"
            landmark.collide = False
            landmark.movable = False
            landmark.boundary = False
            pos = landmark.get_random_pos(world)
            landmark_pos[landmark.name] = pos

        self.landmarks = {landmark.name: landmark for landmark in world.landmarks}

        self.landmark_pos = landmark_pos
        return world

    def reset_world(self, world, random):
        self.num_landmarks = len(self.landmarks)
        self.visited_landmarks = []

        # set random initial states
        for agent in world.agents:
            agent.color = np.array([0, 0, 1])
            agent.state.p_pos = self.np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # set landmarks randomly in the world
        for land_id, landmark in self.landmarks.items():
            if landmark not in world.landmarks:
                world.landmarks.append(landmark)

            if self.landmark_curriculum["current"] == 0:
                landmark.reset(world, position=self.landmark_pos[land_id], size=1)
            elif self.landmark_curriculum["current"] == 1:
                landmark.reset(world, size=1)
            elif self.landmark_curriculum["current"] == 2:
                landmark.reset(world)
            else:
                raise ValueError(
                    f"Value '{self.landmark_curriculum['current']}' has not been implemented for landmark reset"
                )

    # return all agents that are not adversaries
    @staticmethod
    def get_agents(world):
        return [agent for agent in world.agents]

    def reward(self, agent, world):

        def dist_reward():
            min_dist = 99999
            for landmark in world.landmarks:
                dist = get_distance(agent, landmark)
                min_dist = min(min_dist, dist)
                if is_collision(agent, landmark):
                    # positive reward, and add collision
                    self.visited_landmarks.append(landmark.name)

            rew = world.max_size * 2 - min_dist
            rew = min_max_norm(rew, 0, world.max_size * 2)
            return rew

        def collision_reward():
            rew = 0
            for landmark in world.landmarks:
                if is_collision(agent, landmark):
                    # positive reward, and add collision
                    rew += self.landmark_reward
                    self.visited_landmarks.append(landmark.name)

            return rew


        if self.reward_curriculum["current"] == 0:
            rew = dist_reward()
            rew += collision_reward()
            rew = min_max_norm(rew, 0, self.landmark_reward + 1)

        elif self.reward_curriculum["current"] == 1:
            rew = collision_reward()

        elif self.reward_curriculum["current"] == 2:

            rew = self.step_reward
            rew += collision_reward()
            rew = min_max_norm(rew, self.step_reward, self.landmark_reward)


        else:
            raise ValueError(
                f"Value '{self.reward_curriculum['current']}' has not been implemented for reward mode"
            )

        assert 0 <= rew <= 1, f"Reward is not normalized, '{rew}' not in [0,1]"

        return rew

    @staticmethod
    def observation(agent, world):
        return []
