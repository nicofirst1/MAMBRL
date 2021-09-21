import random
import time

from NavigationEnv import get_env


def policy(obs, agent):
    return random.choice(range(3))


env = get_env()

env.reset()
env.render()
for agent in env.agent_iter():
    observation, reward, done, info = env.last()
    action = policy(observation, agent)
    env.step(action)
    env.render()

    time.sleep(0.2)