import numpy as np
from typing import Tuple
from env.timer_landmark import TimerLandmark
from PettingZoo.pettingzoo.mpe._mpe_utils.core import Entity, World, Agent
from PettingZoo.pettingzoo.mpe._mpe_utils.scenario import BaseScenario

def is_collision(entity1, entity2):
    delta_pos = entity1.state.p_pos - entity2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = entity1.size + entity2.size
    return True if dist < dist_min else False


class Border(Entity):
    def __init__(self, start: Tuple[int, int], end: Tuple[int, int], color=(1, 0, 0), linewidth=1):
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
    def __init__(self, num_agents: int, num_landmarks: int, max_size: int, step_reward: int, landmark_reward: int):
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

        self.landmarks = {}
        self.visited_landmarks = []

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
        world.landmarks = [TimerLandmark() for _ in range(self.num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = f"landmark_{i}"
            landmark.collide = False
            landmark.movable = False
            landmark.size = 1
            landmark.boundary = False

        self.landmarks = {landmark.name: landmark for landmark in world.landmarks}
        return world

    def reset_world(self, world, np_random):
        self.num_landmarks = len(self.landmarks)
        self.visited_landmarks = []

        # set random initial states
        for agent in world.agents:
            agent.color = np.array([0, 0, 1])
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # set landmarks randomly in the world
        for landmark in self.landmarks.values():
            if landmark not in world.landmarks:
                world.landmarks.append(landmark)
            landmark.reset(world, np_random)

    # return all agents that are not adversaries
    @staticmethod
    def get_agents(world):
        return [agent for agent in world.agents]

    ## fixme: needed or pettingzoo will not work
    def reward(self, agent, world):
        rew = self.step_reward

        for landmark in world.landmarks:
            if is_collision(agent, landmark):
                # positive reward, and add collision
                rew = self.landmark_reward
                self.visited_landmarks.append(landmark.name)

        return rew

    @staticmethod
    def observation(agent, world):
        return []
