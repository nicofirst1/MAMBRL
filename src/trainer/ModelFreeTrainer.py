from src.agent.PpoWrapper import PpoWrapper
from src.common import Params
from src.env.EnvWrapper import get_env_wrapper
from src.model.ModelFree import ModelFree
from src.trainer.BaseTrainer import BaseTrainer


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
        self.agent = agent(env=self.cur_env, model=model, config=config)

    def set_env(self, new_env):
        self.cur_env = new_env

    def train(self):
        self.agent.set_env(self.cur_env)
        self.agent.learn(epochs=self.config.model_free_epochs)

    # def checkpoint(self):
    # torch.save({
    #     'epoch': EPOCH,
    #     'model_state_dict': net.state_dict(),
    #     'optimizer_state_dict': optimizer.state_dict(),
    #     'loss': LOSS,
    # }, PATH)

    # def restore_training(self):

    # def continue_training(self):


if __name__ == '__main__':
    params = Params()
    env = get_env_wrapper(params)

    trainer = ModelFreeTrainer(ModelFree, PpoWrapper, env, params)
    trainer.train()
