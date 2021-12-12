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

from rich import print

from src.common import Params, get_env_configs
from src.env import get_env

params = Params()
# Get configs
env_config = get_env_configs(params)
env_config['visible'] = True
env_config['obs_shape'] = 320
env = get_env(env_config)

horizon=10
episodes=3
action_dict = {}
done = {}


for ep in range(episodes):
    obs = env.reset()
    done = {_: False for _ in env.agents}
    done["__all__"] = False

    for step in range(horizon):

        if done["__all__"]:
            print("GAME OVER!")
            break

        for agent in env.agents:
            if done[agent]:
                action_dict[agent] = None

            else:
                action_dict[agent] = random.randint(0, 4)
                action_dict[agent] = 4

        obs, reward, done, info = env.step(action_dict)

        output = dict(
            done=done, reward=reward, info={k: v.__dict__ for k, v in info.items()}, obs_dim=obs.shape,
            actions=action_dict,
        )

        print(f"step {step}:\n")
        print(output)

        env.render()

        time.sleep(0.1)

    print(f"End of episode {ep}\n\n\n")

env.close()
