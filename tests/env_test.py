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

from src.common import Params, print_current_strategy
from src.env import get_env, EnvWrapper

params = Params()
# Get configs
env_configs = params.get_env_configs()

env_configs['horizon'] = 15
env_configs['visible'] = True
env_configs['frame_shape'] = [3, 600, 600]

env = EnvWrapper(
    env=get_env(env_configs),
    frame_shape=params.frame_shape,
    num_stacked_frames=1,
    device=params.device,
    gamma=params.gamma
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


def test_landmark_reset_strategy():
    action_dict = {}

    _, landmark_reset_keys, _ = env.get_strategies()

    for land in landmark_reset_keys:

        for ep in range(3):

            env.set_strategy(landmark_reset_strategy=land)

            print_current_strategy(env.get_current_strategy())

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


def test_landmark_collision_strategy(optimal):
    action_dict = {}

    _, _, landmark_collision_keys = env.get_strategies()

    for land in landmark_collision_keys:
        ep = 3
        for ep in range(ep):

            env.set_strategy(landmark_collision_strategy=land)

            print_current_strategy(env.get_current_strategy())

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
                        if optimal:
                            action, action_log_prob = env.optimal_action(agent)
                            action_dict[agent] = action
                        else:
                            action_dict[agent] = random.randint(0, 4)

                obs, reward, done, info = env.step(action_dict)

                time.sleep(0.1)

            print(f"End of episode {ep}\n\n\n")


def test_reward_step_strategy(optimal=False):
    action_dict = {}


    reward_valid_keys, _, _ = env.get_strategies()

    for rew_cur in reward_valid_keys:

        for ep in range(1):

            env.set_strategy(reward_step_strategy=rew_cur)

            print_current_strategy(env.get_current_strategy())

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
                        if optimal:
                            action, action_log_prob = env.optimal_action(agent)
                            action_dict[agent] = action
                        else:
                            action_dict[agent] = random.randint(0, 4)

                obs, reward, done, info = env.step(action_dict)

                time.sleep(0.1)

            print(f"End of episode {ep}\n\n\n")


def test_reward_collision_strategy(optimal=False):
    action_dict = {}

    reward_valid_keys, _, _ = env.get_strategies()

    for rew_cur in reward_valid_keys:

        for ep in range(1):

            env.set_strategy(reward_collision_strategy=rew_cur)

            print_current_strategy(env.get_current_strategy())

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
                        if optimal:
                            action, action_log_prob = env.optimal_action(agent)
                            action_dict[agent] = action
                        else:
                            action_dict[agent] = random.randint(0, 4)

                obs, reward, done, info = env.step(action_dict)

                time.sleep(0.1)

            print(f"End of episode {ep}\n\n\n")


def test_reward_step_strategy_specific(optimal=False):
    action_dict = {}

    env.set_strategy(reward_step_strategy="change_landmark_avoid_borders")

    for ep in range(3):


        print_current_strategy(env.get_current_strategy())

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
                    if optimal:
                        action, action_log_prob = env.optimal_action(agent)
                        action_dict[agent] = action
                    else:
                        action_dict[agent] = random.randint(0, 4)

            obs, reward, done, info = env.step(action_dict)

            time.sleep(0.1)

        print(f"End of episode {ep}\n\n\n")


def test_dones():
    action_dict = {}
    env.set_strategy(landmark_collision_strategy="remove")

    for ep in range(300):

        obs = env.reset()
        done = {_: False for _ in env.agents}
        done["__all__"] = False

        for step in range(env_configs['horizon']):



            for agent in env.agents:
                if done[agent]:
                    action_dict[agent] = None

                else:
                    action_dict[agent] = random.randint(0, 4)
                    # action_dict[agent] = 4

            obs, reward, done, info = env.step(action_dict)

            if done["__all__"]:
                print("GAME OVER!")
                break


def test_optimal_action():
    #test_landmark_collision_strategy(optimal=True)
    #test_reward_step_strategy(optimal=True)
    test_reward_collision_strategy(optimal=True)

