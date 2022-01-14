import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import trange

from logging_callbacks.wandbLogger import preprocess_logs
from src.common import Params
from src.common.schedulers import CurriculumScheduler, GuidedLearningScheduler, LearningRateScheduler, StepScheduler
from model import EnvModelTrainer, MultimodalMAS, PpoWrapper, NextFramePredictor
from src.env import get_env, EnvWrapper

params = Params()

if not params.visible:
    import pyglet

    pyglet.options['shadow_window'] = False


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
        self.env_model = None
        #self.env_model = NextFramePredictor(config)
        #self.env_model = self.env_model.to(self.config.device)

        self.trainer = None
        #self.trainer = EnvModelTrainer(self.env_model, config)

        ## fixme: anche qua bisogna capire se ne serve uno o uno per ogni agente
        self.simulated_env = None
        # self.simulated_env = SimulatedEnvironment(self.real_env, self.env_model, self.action_space, self.config.device)

        self.ppo_wrapper = PpoWrapper(env=self.real_env, config=config)

        if self.config.use_wandb:
            from pytorchCnnVisualizations.src import CamExtractor, ScoreCam

            model = self.ppo_wrapper.actor_critic_dict["agent_0"].base
            if config.base == "resnet":

                cams = []
                for idx, layer in enumerate(list(model.features)):
                    extractor = CamExtractor(model, target_layer=idx)
                    name = type(layer).__name__
                    score_cam = ScoreCam(model, extractor)
                    cams.append(score_cam)
            elif config.base == "cnn":
                cams = []
                for idx, layer in enumerate(list(model.modules())):
                    extractor = CamExtractor(model, target_layer=idx)
                    name = type(layer).__name__
                    score_cam = ScoreCam(model, extractor)
                    cams.append(score_cam)
            else:
                cams = []

            from logging_callbacks import PPOWandb

            self.logger = PPOWandb(
                train_log_step=5,
                val_log_step=5,
                project="model_free",
                opts={},
                models=self.ppo_wrapper.actor_critic_dict["agent_0"].get_modules(),
                horizon=params.horizon,
                mode="disabled" if params.debug else "online",
                action_meaning=self.real_env.env.action_meaning_dict,
                cams=cams,
            )

    def collect_trajectories(self):
        self.ppo_wrapper.set_env(self.real_env)
        agent = RandomAction(self.config.num_actions, self.config.device)

        ## fixme: qui impostasto sempre con doppio ciclo, ma l'altro codice usa un ciclo solo!
        for _ in trange(self.config.episodes, desc="Collecting trajectories.."):
            # init dicts and reset env
            action_dict = {agent_id: False for agent_id in self.real_env.agents}
            done = {agent_id: False for agent_id in self.real_env.env.agents}

            observation = self.real_env.reset()

            for step in range(self.config.horizon):
                observation = observation.unsqueeze(dim=0).to(self.config.device)

                for agent_id in self.real_env.agents:
                    with torch.no_grad():
                        action, _, _ = agent.act(agent_id, observation, full_log_prob=True )
                        action_dict[agent_id] = action

                    if done[agent_id]:
                        action_dict[agent_id]=None

                observation, _, done, _ = self.real_env.step(action_dict)

    def train_agent_sim_env(self, epoch):
        self.ppo_wrapper.set_env(self.simulated_env)

        for _ in trange(1000, desc="Training agent in simulated environment"):
            self.simulated_env.frames = self.simulated_env.get_initial_frame()
            losses = self.ppo_wrapper.learn(episodes=self.config.episodes)

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
        self.ppo_wrapper.set_env(self.real_env)

        for step in trange(1000, desc="Training model free"):
            value_loss, action_loss, entropy, rollout = self.ppo_wrapper.learn(
                episodes=self.config.episodes
            )

            if self.config.use_wandb:
                losses = dict(
                    value_loss=[value_loss],
                    action_loss=[action_loss],
                    entropy=[entropy],
                )
                self.logger.on_batch_end(logs=losses, batch_id=step, rollout=rollout)

    def train_model_free_curriculum(self):
        self.ppo_wrapper.set_env(self.real_env)

        episodes = 3000

        schedulers = init_schedulers(self, episodes,
            use_curriculum=False, use_guided_learning=False, use_learning_rate=False, use_entropy_reg=False
        )

        for step in trange(episodes, desc="Training model free"):
            out = self.ppo_wrapper.learn(
                episodes=self.config.episodes,
            )

            for s in schedulers:
                s.update_step(step)

            if self.config.use_wandb:
                logs, rollout = preprocess_logs(out, self)
                self.logger.on_batch_end(logs=logs, batch_id=step, rollout=rollout)

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


def init_schedulers(mambrl: MAMBRL, episodes, use_curriculum: bool = True, use_guided_learning: bool = True,
                    use_learning_rate: bool = True, use_entropy_reg: bool = True):
    schedulers = []

    if use_curriculum:
        curriculum = {
            400: dict(reward=0, landmark=1),
            600: dict(reward=1, landmark=0),
            800: dict(reward=1, landmark=1),
            900: dict(reward=0, landmark=2),
            1100: dict(reward=1, landmark=2),
            1300: dict(reward=2, landmark=2),
        }

        cs = CurriculumScheduler(values_list=list(curriculum.values()), episodes=episodes,
                                 set_fn=mambrl.real_env.set_curriculum,
                                 step_list=list(curriculum.keys()),
                                 get_curriculum_fn=mambrl.real_env.get_curriculum)
        schedulers.append(cs)

    if use_guided_learning:
        guided_learning = {
            100: 0.8,
            200: 0.7,
            400: 0.6,
            600: 0.4,
            800: 0.2,
            900: 0.0,
            1200: 0.4,
            1400: 0.2,
            1600: 0.1,
            1700: 0.0,
        }

        ep=int(episodes*0.8)
        guided_learning=linear_decay(start_val=1, episodes=ep)

        gls = StepScheduler(values_list=guided_learning, episodes=ep,
                                      set_fn=mambrl.ppo_wrapper.set_guided_learning_prob)
        schedulers.append(gls)

    if use_learning_rate:
        kwargs = dict(gamma=0.997)
        lrs = LearningRateScheduler(base_scheduler=ExponentialLR,
                                    optimizer_dict=mambrl.ppo_wrapper.ppo_agent.optimizers,
                                    scheduler_kwargs=kwargs)
        schedulers.append(lrs)

    if use_entropy_reg:
        values = exponential_decay(Params().entropy_coef, episodes, gamma=0.999)

        es = StepScheduler(
            values_list=values, episodes=episodes,
            set_fn=mambrl.ppo_wrapper.set_entropy_coeff
        )

        schedulers.append(es)

    return schedulers


if __name__ == "__main__":
    params = Params()
    mambrl = MAMBRL(params)
    mambrl.train_env_model()
