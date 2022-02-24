from src.common import Params
from src.agent.RolloutStorage import RolloutStorage
from src.trainer.Policies import TrajCollectionPolicy


class BaseTrainer:
    def __init__(self, env,  config: Params):
        """__init__ method.

        config is a Params object which class is defined in src/common/Params.py
        """
        self.config = config
        self.logger = None
        self.device = config.device

        # wrapper_configs = frame_shape, num_stacked_frames, device, gamma
        wrapper_configs = self.config.get_env_wrapper_configs()
        # env_config = horizon, continuous_actions, gray_scale, frame_shape,
        # visible,
        # scenario_kwargs = step_reward, landmark_reward, landmark_penalty,
        # border_penalty, num_agents, num_landmarks, max_size
        self.cur_env = env
        self.cur_env.set_strategy(**config.strategy)
        # self.obs_shape = self.real_env.obs_shape
        # self.action_space = self.real_env.action_space

        self.policy = TrajCollectionPolicy()

    def collect_trajectories(self) -> RolloutStorage:
        """collect_trajectories method.

        collect trajectories given a policy
        Parameters
        ----------
        policy :
            policy should have a act method
        Returns
        -------
        None.

        """

        raise NotImplementedError("Subclasses should implement this method!!")

    def train(self, rollout: RolloutStorage):
        raise NotImplementedError("Subclasses should implement this method!!")

    def checkpoint(self):
        raise NotImplementedError("Subclasses should implement this method!!")

    def restore_training(self):
        raise NotImplementedError("Subclasses should implement this method!!")
