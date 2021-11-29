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
import random
import time

from rich import print, print_json

from src.common import Params, get_env_configs
from src.env import get_env

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
    if done["__all__"]:
        print("GAME OVER!")
        break

    for agent in env.agents:
        if done[agent]:
            action_dict[agent] = None

        else:
            action_dict[agent] = random.randint(0, 4)

    obs, reward, done, info = env.step(action_dict)

    output = dict(
        done=done,
        reward=reward,
        info={k: v.__dict__ for k, v in info.items()}
    )

    print(f"step {count}:\n")
    print(output)

    time.sleep(0.4)

    count += 1

env.close()
