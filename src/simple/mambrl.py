import torch
import wandb
from tqdm import trange

#from logging_callbacks import PPOWandb
#from logging_callbacks.callbacks import WandbLogger
from ppo_wrapper import PPO
from simulated_env import SimulatedEnvironment
from src.common import get_env_configs, Params
from src.env import get_env
from src.model.EnvModel import NextFramePredictor
from src.train.Policies import MultimodalMAS


class MAMBRL:

    def __init__(self, config):
        self.config = config
        self.logger = None

        self.real_env = get_env(get_env_configs(params))
        self.obs_shape = self.real_env.reset().shape
        self.action_space = self.real_env.action_spaces["agent_0"].n

        ## fixme: per ora c'è solo un env_model, bisogna capire come gestire il multi agent
        self.env_model = NextFramePredictor(config)
        self.env_model = self.env_model.to(self.config.device)

        self.trainer = None

        ## fixme: anche qua bisogna capire se ne serve uno o uno per ogni agente
        self.simulated_env = SimulatedEnvironment(self.real_env, self.env_model, self.action_space, self.config.device)

        self.agent = PPO(
            env=self.simulated_env,
            obs_shape=self.obs_shape,
            action_space=self.action_space,
            num_agents=self.config.agents,
            device=config.device,
            gamma=config.gamma,
            num_steps=self.config.horizon,
            num_minibatch=self.config.minibatch,
            lr=config.lr
        )

        if self.config.use_wandb:
            self.logger= PPOWandb(
                train_log_step=5,
                val_log_step=5,
                project="model_free",
                opts={},
                models={},
                horizon=params.horizon,
                #mode="offline"
            )


    def collect_trajectories(self):
        self.agent.set_env(self.real_env)
        agent = MultimodalMAS(self.agent)

        ## fixme: qui impostasto sempre con doppio ciclo, ma l'altro codice usa un ciclo solo!
        for episode in trange(self.config.episodes, desc="Collecting trajectories.."):
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

        for step in trange(1000, desc="Training agent in simulated environment"):
            initial_frame = self.simulated_env.get_initial_frame()
            zero_frame = torch.zeros((3, 96, 96))

            self.simulated_env.frames = [zero_frame, zero_frame, zero_frame, initial_frame]
            losses = self.agent.learn(episodes=self.config.episodes)

    def train(self):
        for epoch in trange(self.config.epochs, desc="Epoch"):
            self.collect_trajectories()
            #self.trainer.train(epoch, self.real_env)
            self.train_agent_sim_env(epoch)

    def train_env_model(self):
        for step in trange(1000, desc="Training env model"):
            self.collect_trajectories()
            self.trainer.train(step, self.real_env)

    def train_model_free(self):
        self.agent.set_env(self.real_env)

        for step in trange(1000, desc="Training model free"):
            value_loss, action_loss, entropy, rollout = self.agent.learn(episodes=self.config.episodes)

            if self.config.use_wandb:
                losses=dict(value_loss=[value_loss], action_loss=[action_loss], entropy=[entropy])
                self.logger.on_batch_end(logs=losses, batch_id=step,rollout=rollout)

if __name__ == '__main__':
    params = Params()
    mambrl = MAMBRL(params)
    mambrl.train_env_model()
