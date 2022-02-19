import torch
from tqdm import trange

from src.agent.PpoWrapper import PpoWrapper
from src.common import Params
from src.env import EnvWrapper
from src.env.EnvWrapper import get_env_wrapper
from src.env.SimulatedEnv import SimulatedEnvironment
from src.model import NextFramePredictor
from src.model.ModelFree import ModelFree
from src.trainer.BaseTrainer import BaseTrainer
from src.trainer.EnvModelTrainer import EnvModelTrainer
from src.trainer.ModelFreeTrainer import ModelFreeTrainer
from src.trainer.Policies import MultimodalMAS


class FullTrainer(BaseTrainer):
    def __init__(self, config: Params):
        """__init__ module.

        Parameters
        ----------
        model : NextFramePredictor
            model in src.model.EnvModel 
        env : env class
            one of the env classes defined in the src.env directory
        config : Params
            instance of the class Params defined in src.common.Params

        Returns
        -------
        None.

        """

        env = get_env_wrapper(config)

        super(FullTrainer, self).__init__(env, config)


        self.em_trainer = EnvModelTrainer(NextFramePredictor, self.cur_env, config)


        self.mf_trainer = ModelFreeTrainer(ModelFree, PpoWrapper, env, params)

        self.policy = MultimodalMAS(self.mf_trainer.agent.actor_critic_dict)

        self.simulated_env = SimulatedEnvironment(self.cur_env, self.em_trainer.env_model, self.policy, config)

    def train(self):
        for epoch in trange(self.config.epochs, desc="Epoch"):
            self.collect_trajectories()
            self.em_trainer.train(epoch, self.cur_env, steps=20)
            self.train_agent_sim_env()

    def train_agent_sim_env(self):
        self.mf_trainer.set_env(self.simulated_env)
        self.simulated_env.frames = self.simulated_env.get_initial_frame()
        self.mf_trainer.train()


if __name__ == '__main__':
    params = Params()

    trainer = FullTrainer(params)
    trainer.train()
