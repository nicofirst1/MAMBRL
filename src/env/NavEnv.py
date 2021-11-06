import numpy as np
from .Scenario import Scenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from ray.rllib.env import PettingZooEnv
from typing import Dict

def get_env(kwargs):
    """ The env function wraps the environment in 3 wrappers by default. These
    wrappers contain logic that is common to many pettingzoo environments.
    We recommend you use at least the OrderEnforcingWrapper on your own environment
    to provide sane error messages. You can find full documentation for these methods
    elsewhere in the developer documentation. """

    env = make_env(raw_env)
    env = env(env_context=kwargs)
    # env=to_parallel(env)
    env = PettingZooEnv(env)
    return env

class raw_env(SimpleEnv):
    def __init__(self, env_context: Dict):
        scenario = Scenario()
        world = scenario.make_world(env_context["N"], env_context["landmarks"])
        super().__init__(
            scenario,
            world,
            env_context["max_cycles"],
            env_context["continuous_actions"],
            #special_entities=TimerLandmark,
        )
        self.metadata["name"] = "collab_nav"

    @staticmethod
    def is_collision(agent1, agent2):
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