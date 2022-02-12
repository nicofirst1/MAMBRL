import torch
from .BaseTrainer import BaseTrainer


class ModelFreeTrainer(BaseTrainer):
    def __init__(self, model, agent, env, config):
        """__init__ method


        Parameters
        ----------
        model : ModelFree class
            one of the ModelFree classes defined in the src.model directory
        agent : agent class
            one of the agent classes defined in the src.agent directory
        env : env class
            one of the env classes defined in the src.env directory
        config : Params
            instance of the class Params defined in src.common.Params

        Returns
        -------
        None.

        """
        super(ModelFreeTrainer, self).__init__(env, config)

        self.agent = agent(env=self.real_env, model=model, config=config)

    def train(self):
        self.real_env.set_strategy(**self.config.strategy)
        self.agent.set_env(self.real_env)
        self.agent.learn(epochs=self.config.epochs)

    # def checkpoint(self):
        # torch.save({
        #     'epoch': EPOCH,
        #     'model_state_dict': net.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': LOSS,
        # }, PATH)

    # def restore_training(self):

    # def continue_training(self):
