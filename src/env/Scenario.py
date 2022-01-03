from typing import List, Tuple

import numpy as np

from PettingZoo.pettingzoo.mpe._mpe_utils.core import Entity, World, Agent
from PettingZoo.pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from .TimerLandmark import TimerLandmark


def is_collision(agent1, agent2):
    delta_pos = agent1.state.p_pos - agent2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = agent1.size + agent2.size
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

    # @property
    # def contact_margin(self):
    #     if len(self.entities) == 0:
    #         return self._contact_margin
    #     else:
    #         min_size = [ent.size for ent in self.entities]
    #         min_size = min(min_size)
    #         return min_size*100
    #
    # @contact_margin.setter
    # def contact_margin(self, value):
    #
    #     self._contact_margin = value


class Scenario(BaseScenario):
    def __init__(
            self,
            num_agents: int,
            num_landmarks: int,
            max_size: int,
            landmark_reward: int = 1,
            max_landmark_counter: int = 4,
            landmark_penalty: int = 2,
    ):
        """

        Args:
            num_agents: the number of agents in the system
            num_landmarks: the number of landmarks
            max_size: maximum size of the word, used in word initialization
            landmark_reward: the reward value (positive) for visiting a landmark
            max_landmark_counter: At each timestep a landmarks increases its penalty until step>=max_landmark_counter,
                    then it stays the same until visited
            landmark_penalty: the penalty value (negative) for not visited landmarks
        """
        self.registered_collisions = {}
        self.landmarks = {}
        self.max_landmark_counter = max_landmark_counter
        self.landmark_reward = landmark_reward
        self.landmark_penalty = landmark_penalty
        self.num_agents = num_agents
        self.num_landmarks = num_landmarks
        self.max_size = max_size

    def get_reward_range(self) -> List[int]:
        """
        Return the possible values of the reward
        """

        lower_bound = (
                self.num_landmarks * self.landmark_penalty * self.max_landmark_counter
        )
        upper_bound = self.landmark_reward + 1

        return list(range(lower_bound, upper_bound))

    def make_world(
            self,
    ) -> World:
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
        self.registered_collisions = {agent.name: [] for agent in world.agents}

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

        self.registered_collisions = {agent.name: [] for agent in world.agents}

        # set random initial states
        for agent in world.agents:
            agent.color = np.array([0, 0, 1])
            agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        # set landmarks randomly in the world
        for i, landmark in enumerate(world.landmarks):
            landmark.reset(world, np_random)

    # return all agents that are not adversaries
    @staticmethod
    def get_agents(world):
        return [agent for agent in world.agents]

    def reward(self, agent, world):
        """
        Rewards are now bounded between
        [- num_landmarks * landmark_penalty * max_landmark_counter ; landmark_reward]
        Assuming only integer values
        """
        rew = 0

        # check for every landmark
        for landmark in world.landmarks:

            already_collided = landmark.name in self.registered_collisions[agent.name]
            had_collided = is_collision(agent, landmark)

            if not already_collided and had_collided:
                # positive reward, and add collision
                rew += self.landmark_reward
                self.registered_collisions[agent.name] += [landmark.name]
            elif already_collided and not had_collided:
                # if not on landmark and remove registered collision
                self.registered_collisions[agent.name].remove(landmark.name)

            rew += self.landmark_penalty * min(
                landmark.counter, self.max_landmark_counter
            )

        return rew

    @staticmethod
    def observation(agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)

        # communication of all other agents
        comm = []
        other_agents_pos = []
        other_agents_vel = []
        for other in world.agents:
            comm.append(other.state.c)
            other_agents_pos.append(other.state.p_pos - agent.state.p_pos)
            other_agents_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel] +
            [agent.state.p_pos] +
            entity_pos +
            other_agents_pos +
            other_agents_vel
        )
