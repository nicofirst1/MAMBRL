import torch
from rich.progress import track
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange
import keyboard

from logging_callbacks.wandbLogger import preprocess_logs
from src.common import Params
from src.common.schedulers import CurriculumScheduler, LearningRateScheduler, StepScheduler, \
    linear_decay, exponential_decay
from src.env import get_env, EnvWrapper
from src.model import EnvModelTrainer, NextFramePredictor
from src.ppo.PpoWrapper import PpoWrapper
from src.train.Policies import OptimalAction

params = Params()

if not params.visible:
    import pyglet
    pyglet.options['shadow_window'] = False


class MAMBRL:
    def __init__(self, config: Params):
        """__init__ method.

        config is a Params object which class is defined in
        src/common/Params.py
        """
        self.config = config
        self.logger = None

        # wrapper_configs = frame_shape, num_stacked_frames, device, gamma
        wrapper_configs = self.config.get_env_wrapper_configs()
        # env_config = horizon, continuous_actions, gray_scale, frame_shape,
        # visible,
        # scenario_kwargs = step_reward, landmark_reward, landmark_penalty,
        # border_penalty, num_agents, num_landmarks, max_size
        self.real_env = EnvWrapper(
            env=get_env(self.config.get_env_configs()),
            **wrapper_configs,
        )

        self.obs_shape = self.real_env.obs_shape
        self.action_space = self.real_env.action_space

        # fixme: per ora c'è solo un env_model, bisogna capire come gestire il multi agent
        self.env_model = NextFramePredictor(config)
        self.env_model = self.env_model.to(self.config.device)

        self.trainer = EnvModelTrainer(self.env_model, config)

        # fixme: anche qua bisogna capire se ne serve uno o uno per ogni agente
        self.simulated_env = None
        # self.simulated_env = SimulatedEnvironment(self.real_env, self.env_model, self.action_space, self.config.device)

        self.ppo_wrapper = PpoWrapper(env=self.real_env, config=config)

    def collect_trajectories(self):
        # FIXME: perché settiamo di nuovo l'env del PPOWrapper, non dovrebbe già
        # essere settato nell' __init__?
        self.ppo_wrapper.set_env(self.real_env)
        agent = OptimalAction(
            self.real_env, self.config.num_actions, self.config.device)
        # fixme: qui impostasto sempre con doppio ciclo, ma l'altro codice usa un ciclo solo!
        for _ in track(range(self.config.episodes), description="Collecting trajectories..", total=self.config.episodes):
            # init dicts and reset env
            action_dict = {
                agent_id: False for agent_id in self.real_env.agents}
            done = {agent_id: False for agent_id in self.real_env.env.agents}
            done["__all__"] = False
            observation = self.real_env.reset()

            for step in range(self.config.horizon):
                observation = observation.unsqueeze(
                    dim=0).to(self.config.device)

                for agent_id in self.real_env.agents:
                    with torch.no_grad():
                        action, _, _ = agent.act(
                            agent_id, observation, full_log_prob=True)
                        action_dict[agent_id] = action

                    if done[agent_id]:
                        action_dict[agent_id] = None
                    if done["__all__"]:
                        break

                observation, _, done, _ = self.real_env.step(action_dict)
                if done["__all__"]:
                    break

    def train_agent_sim_env(self, epoch):
        self.ppo_wrapper.set_env(self.simulated_env)
        self.simulated_env.frames = self.simulated_env.get_initial_frame()
        self.ppo_wrapper.learn(episodes=self.config.episodes)

    def train(self):
        for epoch in trange(self.config.epochs, desc="Epoch"):
            self.collect_trajectories()
            # self.trainer.train(epoch, self.real_env)
            self.train_agent_sim_env(epoch)

    def train_env_model(self):
        epochs = 3000
        for step in trange(epochs, desc="Training env model"):
            self.collect_trajectories()
            self.trainer.train(step, self.real_env)

    def train_model_free(self):

        # self.real_env.set_strategy(reward_step_strategy="positive_distance")
        self.real_env.set_strategy(reward_step_strategy="time_penalty")
        self.ppo_wrapper.set_env(self.real_env)
        self.ppo_wrapper.learn(epochs=self.config.epochs)

    def user_game(self):
        moves = {"w": 4, "a": 1, "s": 3, "d": 2}

        game_reward = 0
        finish_game = False

        self.real_env.reset()
        while finish_game is False:
            while True:
                self.real_env.env.render()
                if keyboard.is_pressed("w"):
                    user_move = 4
                    break
                elif keyboard.is_pressed("a"):
                    user_move = 1
                    break
                elif keyboard.is_pressed("s"):
                    user_move = 3
                    break
                elif keyboard.is_pressed("d"):
                    user_move = 2
                    break

            _, reward, done, _ = self.real_env.step({"agent_0": user_move})
            game_reward += reward["agent_0"]

            if done["__all__"]:
                while True:
                    print("Finee! Total reward: ", game_reward)
                    exit_input = input(
                        "Gioco terminato! Iniziare un'altra partita? (y/n)"
                    )
                    if exit_input == "n":
                        finish_game = True
                        self.real_env.env.close()
                        break
                    elif exit_input == "y":
                        game_reward = 0
                        self.real_env.reset()
                        break


if __name__ == "__main__":
    params = Params()
    mambrl = MAMBRL(params)
    mambrl.train_model_free()
