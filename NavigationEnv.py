from typing import Dict

import matplotlib as mpl
import numpy as np
from pettingzoo.mpe._mpe_utils.core import Agent, World, Entity
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from ray.rllib.env import PettingZooEnv


def get_env(kwargs):
    '''
    The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation.
    '''
    env = make_env(raw_env)
    env = env(env_context=kwargs)
    # env=to_parallel(env)
    env = PettingZooEnv(env)
    return env


def colorFader(c1, c2, mix=0):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    return mpl.colors.to_rgb(mpl.colors.to_hex((1 - mix) * c1 + mix * c2))


class TimerLandmark(Entity):
    """
    Timer landmark class.
    Each landmark increases its timer by the 'increase' param per step.
    This timer is proportional (timer* penalty) to the penalty agents get at each turn.
    The timer resets when an agent lands upon it and the timer starts from zero.

    So the longer a landmark stays untouched the worse the penalty gets.
    """

    colors = [colorFader(np.array([0, 1, 0]), np.array([1, 0, 0]), x / 100) for x in range(0, 100)]

    def __init__(self, increase=0.1):
        super().__init__()
        self.timer = 0
        self.increase = increase
        self.counter = 0

    def reset(self, world, np_random):
        self.timer = 0
        self.color = np.array([0, 1.0, 0])
        self.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
        self.state.p_vel = np.zeros(world.dim_p)

    def step(self):
        self.counter += 1
        self.timer = self.increase * self.counter

        if self.counter >= len(self.colors):
            self.color = self.colors[-1]
        else:
            self.color = self.colors[self.counter]

    def reset_timer(self):
        self.timer = 0
        self.counter = 0


class Scenario(BaseScenario):
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
    def get_agents(self, world):
        return [agent for agent in world.agents]

    def reward(self, agent, world):
        rew = 0

        # check for every landmark
        for landmark in world.landmarks:
            rew -= self.std_penalty * landmark.timer

        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # communication of all other agents
        comm = []
        other_pos = []
        other_vel = []
        for other in world.agents:
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
            other_vel.append(other.state.p_vel)
        return np.concatenate(
            [agent.state.p_vel]
            + [agent.state.p_pos]
            + entity_pos
            + other_pos
            + other_vel
        )


class raw_env(SimpleEnv):
    def __init__(self, env_context: Dict):
        scenario = Scenario()
        world = scenario.make_world(env_context['N'], env_context["landmarks"])
        super().__init__(scenario, world, env_context["max_cycles"], env_context["continuous_actions"])
        self.metadata["name"] = "collab_nav"

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def step(self, action):
        super(raw_env, self).step(action)

        for landmark in self.world.landmarks:

            visited = False

            # check if one agent has visited the landmark
            for agent in self.world.agents:
                if self.is_collision(agent, landmark):
                    visited = True
                    break

            # if visited reset else increase
            if visited:
                landmark.reset_timer()
            else:
                landmark.step()
