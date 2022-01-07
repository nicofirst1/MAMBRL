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

from common.utils import print_current_curriculum
from env.env_wrapper import EnvWrapper
from src.common import Params
from src.env import get_env

params = Params()
# Get configs
env_configs= params.get_env_configs()

env_configs['horizon'] = 10
env_configs['visible'] = True
env_configs['frame_shape'] = [3,250,250]

env = EnvWrapper(
    env=get_env(env_configs),
    frame_shape=params.frame_shape,
    num_stacked_frames=params.num_frames,
    device=params.device
)



def test_episodes():
    action_dict = {}
    for ep in range(30):
        obs = env.reset()
        done = {_: False for _ in env.agents}
        done["__all__"] = False

        for step in range(env_configs['horizon']):

            if done["__all__"]:
                print("GAME OVER!")
                break

            for agent in env.agents:
                if done[agent]:
                    action_dict[agent] = None

                else:
                    action_dict[agent] = random.randint(0, 4)
                    # action_dict[agent] = 4

            obs, reward, done, info = env.step(action_dict)

            output = dict(
                done=done, reward=reward, info={k: v.__dict__ for k, v in info.items()}, obs_dim=obs.shape,
                actions=action_dict,
            )

            print(f"step {step}:\n")
            print(output)

            time.sleep(0.1)

        print(f"End of episode {ep}\n\n\n")


def test_landmark_curriculum():
    action_dict = {}

    for land in range(3):

        for ep in range(3):

            env.set_curriculum(landmark=land)

            print_current_curriculum(env.get_curriculum())

            obs = env.reset()
            done = {_: False for _ in env.agents}
            done["__all__"] = False

            for step in range(env_configs['horizon']):

                if done["__all__"]:
                    print("GAME OVER!")
                    break

                for agent in env.agents:
                    if done[agent]:
                        action_dict[agent] = None

                    else:
                        action_dict[agent] = random.randint(0, 4)
                        # action_dict[agent] = 4

                obs, reward, done, info = env.step(action_dict)

                time.sleep(0.1)

            print(f"End of episode {ep}\n\n\n")


def test_reward_curriculum():
    action_dict = {}

    for land in range(3):

        for ep in range(3):

            env.set_curriculum(reward=land)

            print_current_curriculum(env.get_curriculum())

            obs = env.reset()
            done = {_: False for _ in env.agents}
            done["__all__"] = False

            for step in range(env_configs['horizon']):

                if done["__all__"]:
                    print("GAME OVER!")
                    break

                for agent in env.agents:
                    if done[agent]:
                        action_dict[agent] = None

                    else:
                        action_dict[agent] = random.randint(0, 4)
                        # action_dict[agent] = 4

                obs, reward, done, info = env.step(action_dict)

                time.sleep(0.1)

            print(f"End of episode {ep}\n\n\n")



