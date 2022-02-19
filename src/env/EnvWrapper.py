from typing import Tuple

import torch

from src.common.utils import one_hot_encode
from .NavEnv import NavEnv, get_env


class EnvWrapper:
    """EnvWrapper class.

    It provides a wrapper for the Navigation Environment that allows to stack
    multiple frames from the environment by using a buffer
    """

    def __init__(
            self, env: NavEnv, frame_shape, num_stacked_frames, device, gamma
    ):

        self.env = env
        self.buffer = []
        self.gamma = gamma
        self.initial_frame = None
        self.channel_size = frame_shape[0]
        self.stacked_frames = torch.zeros(
            self.channel_size * num_stacked_frames, *frame_shape[1:]
        )

        self.obs_shape = self.stacked_frames.shape
        self.action_space = self.env.action_spaces["agent_0"].n

        self.agents = self.env.agents_dict
        self.device = device

    def set_strategy(self, **kwargs):
        self.env.set_strategy(**kwargs)

    def get_current_strategy(self) -> Tuple[str, str, str, str]:
        """get_current_strategy method.

        return a list of the currently used strategies
        Returns
        -------
        Tuple[str, str, str, str]

        """
        return self.env.get_current_strategy()

    def get_strategies(self):
        """get_current_strategy method.

        returns a list of strings describing the strategies adopted for
        defining certain events in the environment, i.e. reward_step_strategy
        (which reward is given during a normal step in the environment),
        reward_collision_strategy (which reward is given during a collision),
        landmark_reset_strategy (how landmarks are handled),
        landmark_collision_strategy (what happens when there is a
        collision with a landmark)

        Returns
        -------
        Tuple[str, str, str, str]
        """
        return self.env.get_strategies()

    def reset(self):
        observation = self.env.reset()

        num_frames = self.stacked_frames.shape[0] // self.channel_size
        obs = observation.repeat(num_frames, 1, 1)

        self.stacked_frames = obs
        if self.initial_frame is None:
            self.initial_frame = self.stacked_frames.clone()

        return self.stacked_frames

    def step(self, actions):
        new_obs, rewards, done, infos = self.env.step(actions)
        if self.train_env:
            self.add_interaction(actions["agent_0"], torch.tensor(rewards["agent_0"]), new_obs, done["__all__"])

        self.stacked_frames = torch.cat((self.stacked_frames[self.channel_size:], new_obs), dim=0)

        if self.train_env and done["__all__"]:
            value = torch.tensor(0.).to(self.device)
            self.buffer[-1][5] = value
            index = len(self.buffer) - 2
            while reversed(range(len(self.buffer) - 1)):
                # value = (self.buffer[index][2] - 1).to(self.device) + 0.998 * value
                value = self.buffer[index][2] + self.gamma * value
                self.buffer[index][5] = value
                index -= 1

                if self.buffer[index][4] == 1:
                    break

        return self.stacked_frames, rewards, done, infos

    def optimal_action(self, agent):
        return self.env.optimal_action(agent)

    def add_interaction(self, actions, rewards, new_obs, done):
        current_obs = self.stacked_frames.squeeze().byte().cpu()

        if actions is None:
            actions = -1

        action = one_hot_encode(actions, self.action_space).cpu()
        reward = (rewards.squeeze() + 1).byte().cpu()
        new_obs = new_obs.squeeze().byte().cpu()
        done = torch.tensor(done, dtype=torch.uint8).cpu()
        self.buffer.append([current_obs, action, reward, new_obs, done, None])


def get_env_wrapper(params):
    """
    Initialize env wrapper and set strategies
    @param params:
    @return: EnvWrapper
    """
    wrapper_configs = params.get_env_wrapper_configs()

    env = EnvWrapper(
        env=get_env(params.get_env_configs()),
        **wrapper_configs,
    )
    env.set_strategy(**params.strategy)

    return env
