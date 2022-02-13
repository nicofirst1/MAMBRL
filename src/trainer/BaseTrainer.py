from src.env import get_env
from rich.progress import track
import torch


class BaseTrainer:
    def __init__(self, env, config):
        """__init__ method.

        config is a Params object which class is defined in src/common/Params.py
        """
        self.config = config
        self.logger = None

        # wrapper_configs = frame_shape, num_stacked_frames, device, gamma
        wrapper_configs = self.config.get_env_wrapper_configs()
        # env_config = horizon, continuous_actions, gray_scale, frame_shape,
        # visible,
        # scenario_kwargs = step_reward, landmark_reward, landmark_penalty,
        # border_penalty, num_agents, num_landmarks, max_size
        self.real_env = env

        self.obs_shape = self.real_env.obs_shape
        self.action_space = self.real_env.action_space

        # fixme: anche qua bisogna capire se ne serve uno o uno per ogni agente
        self.simulated_env = None
        # self.simulated_env = SimulatedEnvironment(self.real_env, self.env_model, self.action_space, self.config.device)

    def train(self):
        raise NotImplementedError("Subclasses should implement this method!!")

    def checkpoint(self):
        raise NotImplementedError("Subclasses should implement this method!!")

    def restore_training(self):
        raise NotImplementedError("Subclasses should implement this method!!")
