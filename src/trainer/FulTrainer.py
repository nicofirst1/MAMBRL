import torch
from tqdm import trange

from src.agent.PpoWrapper import PpoWrapper
from src.common import Params
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


        self.actor_c=ModelFree()

    def train(self):
        for epoch in trange(self.config.epochs, desc="Epoch"):
            self.em_trainer.train()
            self.train_agent_sim_env()

    def train_agent_sim_env(self):

        for episodes in range(3):
            frames = self.simulated_env.reset()

            for step in range(4):
                actions = {}
                for agent in self.cur_env.agents.keys():
                    actions[agent] = self.simulated_env.get_actions(agent, frames)
                frames, rewards, done, envmodel_features = self.simulated_env.step(actions)
                envmodel_features= envmodel_features[-1]
                envmodel_features= envmodel_features.unsqueeze(dim=0)

                mask=torch.ones(1)
                frames=frames.unsqueeze(dim=0)

                mask = mask.to(self.config.device)
                frames = frames.to(self.config.device)

                modelfree_features={}
                for agent in self.cur_env.agents.keys():
                    modelfree_features[agent]=self.mf_trainer.agent.actor_critic_dict[agent].feature_extraction(frames,mask)

                features={}
                for agent in self.cur_env.agents.keys():
                    features[agent]=torch.cat((modelfree_features[agent], envmodel_features),dim=-1)

                actions, values=1,2
                a=1




if __name__ == '__main__':
    params = Params()

    trainer = FullTrainer(params)
    trainer.train()
