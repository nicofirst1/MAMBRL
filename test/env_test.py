"""env_test file.

Run and visualize the main environment through random actions.

Env main params
---------------
    action: int
        an int between 0 and 4
    observation: array
        700x700x3 array of int values between 0 and 255
    reward: float

"""
import sys
import random
import time
import os
project_dir = os.path.abspath(os.path.join(os.path.curdir,
                                           os.path.pardir))
sys.path.insert(0, project_dir)
from src.env.NavEnv import get_env
from src.common.utils import *

params = Params()
# Get configs
env_config = get_env_configs(params)
env = get_env(env_config)
obs = env.reset()
count = 0
action_dict = {}
done = {}
done = {_: False for _ in env.agents}
done["__all__"] = False

while count <= 1000:
    env.render()
    if done["__all__"] is True:
        print("GAME OVER!")
        break
    for agent in env.agents:
        if done[agent] is False:
            action_dict[agent] = random.randint(0, 4)
        else:
            action_dict[agent] = None
    obs, reward, done, info = env.step(action_dict)
    print(f"step {count}:")
    print(done)
    print(reward)
    print("Press a CTRL+C to continue...")
    try:
        while True:
            env.render("human")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    count += 1

env.close()
