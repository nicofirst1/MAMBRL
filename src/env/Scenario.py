import numpy as np

from src.env.TimerLandmark import TimerLandmark

from pettingzoo.mpe._mpe_utils.core import Agent, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario


class Scenario(BaseScenario):
    std_penalty: int

    def make_world(self, num_agents=1, num_landmarks=2, std_penalty=2):
        world = World()
        # set any world properties first
        world.dim_c = 2
        # add agents
        world.agents = [Agent() for _ in range(num_agents)]
        self.std_penalty = std_penalty

        for i, agent in enumerate(world.agents):
            agent.name = f"agent_{i}"
            agent.collide = True
            agent.silent = True
            agent.size = 0.05
            agent.accel = 4.0
            agent.max_speed = 1.3

        # add landmarks
        world.landmarks = [TimerLandmark() for _ in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = False
            landmark.movable = False
            landmark.size = 0.1
            landmark.boundary = False

        return world

    def reset_world(self, world, np_random):
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
        rew = 0

        # check for every landmark
        for landmark in world.landmarks:
            rew -= self.std_penalty * landmark.timer

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
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_agents_pos
            + other_agents_vel
        )
