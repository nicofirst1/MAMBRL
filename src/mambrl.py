import torch
from tqdm import trange

from src.common import Params

params = Params()

if not params.visible:
    import pyglet

    pyglet.options['shadow_window'] = False

from common.utils import print_current_curriculum
from env.env_wrapper import EnvWrapper
from model.env_model_trainer import EnvModelTrainer
from model.policies import MultimodalMAS
from model.ppo_wrapper import PPO
from src.env import get_env

from src.model.env_model import NextFramePredictor


class MAMBRL:
    def __init__(self, config):
        self.config = config
        self.logger = None

        self.real_env = EnvWrapper(
            env=get_env(self.config.get_env_configs()),
            frame_shape=self.config.frame_shape,
            num_stacked_frames=self.config.num_frames,
            device=self.config.device,
        )

        self.obs_shape = self.real_env.obs_shape
        self.action_space = self.real_env.action_space

        ## fixme: per ora c'è solo un env_model, bisogna capire come gestire il multi agent
        self.env_model = NextFramePredictor(config)
        self.env_model = self.env_model.to(self.config.device)

        self.trainer = EnvModelTrainer(self.env_model, config)

        ## fixme: anche qua bisogna capire se ne serve uno o uno per ogni agente
        self.simulated_env = None
        # self.simulated_env = SimulatedEnvironment(self.real_env, self.env_model, self.action_space, self.config.device)

        self.agent = PPO(
            env=self.real_env,
            obs_shape=self.obs_shape,
            action_space=self.action_space,
            num_agents=self.config.agents,
            config=config,
        )

        if self.config.use_wandb:
            from pytorchCnnVisualizations.src import CamExtractor, LayerCam, ScoreCam

            model = self.agent.actor_critic_dict["agent_0"].base
            cams = []
            if config.base == "resnet":

                target_layer = 7

            elif config.base == "cnn":
                target_layer = 5
            else:
                target_layer = 1

            extractor = CamExtractor(model, target_layer=target_layer)
            layer = LayerCam(model, extractor)
            score_cam = ScoreCam(model, extractor)

            cams = [layer, score_cam]
            from logging_callbacks import PPOWandb

            self.logger = PPOWandb(
                train_log_step=5,
                val_log_step=5,
                project="model_free",
                opts={},
                models={},
                horizon=params.horizon,
                mode="disabled" if params.debug else "online",
                action_meaning=self.real_env.env.action_meaning_dict,
                cams=cams,
            )

    def collect_trajectories(self):
        self.agent.set_env(self.real_env)
        agent = MultimodalMAS(self.agent)

        ## fixme: qui impostasto sempre con doppio ciclo, ma l'altro codice usa un ciclo solo!
        for _ in trange(self.config.episodes, desc="Collecting trajectories.."):
            # init dicts and reset env
            action_dict = {agent_id: False for agent_id in self.real_env.agents}

            observation = self.real_env.reset()

            for step in range(self.config.horizon):
                observation = observation.unsqueeze(dim=0).to(self.config.device)

                for agent_id in self.real_env.agents:
                    with torch.no_grad():
                        action, _, _ = agent.act(agent_id, observation)
                        action_dict[agent_id] = action

                observation, _, _, _ = self.real_env.step(action_dict)

    def train_agent_sim_env(self, epoch):
        self.agent.set_env(self.simulated_env)

        for _ in trange(1000, desc="Training agent in simulated environment"):
            self.simulated_env.frames = self.simulated_env.get_initial_frame()
            losses = self.agent.learn(episodes=self.config.episodes)

    def train(self):
        for epoch in trange(self.config.epochs, desc="Epoch"):
            self.collect_trajectories()
            # self.trainer.train(epoch, self.real_env)
            self.train_agent_sim_env(epoch)

    def train_env_model(self):
        for step in trange(1000, desc="Training env model"):
            self.collect_trajectories()
            self.trainer.train(step, self.real_env)

    def train_model_free(self):
        self.agent.set_env(self.real_env)

        for step in trange(1000, desc="Training model free"):
            value_loss, action_loss, entropy, rollout = self.agent.learn(
                episodes=self.config.episodes, full_log_prob=True
            )

            if self.config.use_wandb:
                losses = dict(
                    value_loss=[value_loss],
                    action_loss=[action_loss],
                    entropy=[entropy],
                )
                self.logger.on_batch_end(logs=losses, batch_id=step, rollout=rollout)

    def train_model_free_curriculum(self):
        self.agent.set_env(self.real_env)

        episodes = 1200

        curriculum = {
            400: dict(reward=0, landmark=1),
            600: dict(reward=1, landmark=0),
            800: dict(reward=1, landmark=1),
            1000: dict(reward=1, landmark=2),
            1200: dict(reward=2, landmark=2),
        }

        for step in trange(episodes, desc="Training model free"):
            value_loss, action_loss, entropy, rollout = self.agent.learn(
                episodes=self.config.episodes, full_log_prob=True, #entropy_coef=1 / (step + 1)
            )

            if self.config.use_wandb:
                losses = dict(
                    value_loss=[value_loss],
                    action_loss=[action_loss],
                    entropy=[entropy],
                )
                self.logger.on_batch_end(logs=losses, batch_id=step, rollout=rollout)
            if step in curriculum.keys():
                self.real_env.set_curriculum(**curriculum[step])
                self.real_env.get_curriculum()
                print_current_curriculum(self.real_env.get_curriculum())

    def user_game(self):
        moves = {"w": 4, "a": 1, "s": 3, "d": 2}

        game_reward = 0
        finish_game = False

        self.real_env.reset()
        while finish_game is False:
            user_input = str(input())

            try:
                user_move = moves[user_input]
            except KeyError:
                continue

            _, reward, done, _ = self.real_env.step({"agent_0": user_move})
            game_reward += reward["agent_0"]

            if done:
                while True:
                    print("Finee! Total reward: ", game_reward)
                    exit_input = input(
                        "Gioco terminato! Iniziare un'altra partita? (y/n)"
                    )
                    if exit_input == "n":
                        finish_game = True
                        break
                    elif exit_input == "y":
                        game_reward = 0
                        self.real_env.reset()
                        break


if __name__ == "__main__":
    params = Params()
    mambrl = MAMBRL(params)
    mambrl.train_model_free_curriculum()
