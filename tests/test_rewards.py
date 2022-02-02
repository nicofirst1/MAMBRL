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
import numpy as np

from rich import print

from src.common.Params import Params
from src.env.NavEnv import get_env

params = Params()
# Get configs
env_config = params.get_env_configs()
env_config["visible"] = True
env_config["frame_shape"] = [3, 600, 600]
# env_config["scenario_kwargs"]
env = get_env(env_config)

horizon = params.horizon
episodes = 1  # params.episodes
action_dict = {}
done = {}


for ep in range(episodes):
    landmarks_positions = np.array([[-1.0, 0.0], [1.0, 0.0]])
    agents_positions = np.array([[0.0, 0.0]])
    strategy = dict(reward_step_strategy="change_landmark",
                    reward_collision_strategy="change_landmark_avoid_borders",
                    landmark_reset_strategy="simple",
                    landmark_collision_strategy="stay")
    env.set_strategy(**strategy)
    # landmarks_positions=landmarks_positions,
    obs = env.reset(landmarks_positions=landmarks_positions,
                    agents_positions=agents_positions)
    # agents_positions=agents_positions)
    done = {_: False for _ in env.agents}
    done["__all__"] = False
    counter = 0
    total_reward = 0.0
    for step in range(horizon):

        if done["__all__"]:
            print("GAME OVER!")
            break

        for agent in env.agents:
            if done[agent]:
                action_dict[agent] = None

            else:
                # action_dict[agent] = random.randint(0, 4)
                # 0 don't move, 1 left, 2 right,  3 down, 4 top
                if counter < 6:
                    action_dict[agent] = 1
                elif counter >= 6 and counter <= 9:
                    action_dict[agent] = 2
                elif counter > 9 and counter <= 12:
                    action_dict[agent] = 1
                # elif counter >= 6 and counter <= 18:
                #     action_dict[agent] = 2
                # elif counter > 18 and counter <= 30:
                #     action_dict[agent] = 1
                else:
                    counter = 5

        obs, reward, done, info = env.step(action_dict)
        env.steps

        output = dict(
            done=done, reward=reward, info={k: v.__dict__ for k, v in info.items()}, obs_dim=obs.shape,
            actions=action_dict,
        )

        print(f"step {step}:")
        print(f"env steps {env.steps}:")
        print(output["reward"])
        print("\n")

        total_reward += output["reward"]["agent_0"]

        env.render()
        counter += 1
        time.sleep(0.3)

    print(f"End of episode {ep}\n\n\n")
    print(f"Episode total reward = {total_reward}")
    print(f"Episode mean reward = {total_reward/horizon}")
env.close()
