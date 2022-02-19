from tqdm import trange

from src.env import get_env
from rich.progress import track
import torch

from src.trainer.Policies import TrajCollectionPolicy


class BaseTrainer:
    def __init__(self, env,  config):
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
        self.cur_env = env

        # self.obs_shape = self.real_env.obs_shape
        # self.action_space = self.real_env.action_space

        self.policy= TrajCollectionPolicy()

    def collect_trajectories(self):
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


        # fixme: qui impostasto sempre con doppio ciclo, ma l'altro codice usa un ciclo solo!
        for _ in trange(self.config.episodes, desc="Collecting trajectories.."):
            # init dicts and reset env
            action_dict = {
                agent_id: False for agent_id in self.cur_env.agents}
            done = {agent_id: False for agent_id in self.cur_env.env.agents}
            done["__all__"] = False
            observation = self.cur_env.reset()

            for step in range(self.config.horizon):
                observation = observation.unsqueeze(
                    dim=0).to(self.config.device)

                for agent_id in self.cur_env.agents:
                    with torch.no_grad():
                        action, _, _ = self.policy.act(
                            agent_id, observation)
                        action_dict[agent_id] = action

                    if done[agent_id]:
                        action_dict[agent_id] = None
                    if done["__all__"]:
                        break

                observation, _, done, _ = self.cur_env.step(action_dict)
                if done["__all__"]:
                    break

    def train(self, agent):
        raise NotImplementedError("Subclasses should implement this method!!")

    def checkpoint(self):
        raise NotImplementedError("Subclasses should implement this method!!")

    def restore_training(self):
        raise NotImplementedError("Subclasses should implement this method!!")
