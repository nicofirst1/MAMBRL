"""tensorboard_test file."""
import os
from torch.utils.tensorboard import SummaryWriter
from src.model.ModelFree import ModelFree
from src.env.NavEnv import get_env
from src.common.utils import *

TENSORBOARD_DIR = os.path.join(os.path.abspath(os.pardir), "tensorboard")

params = Params()
env_config = get_env_configs(params)
env = get_env(env_config)
obs_space = env.render(mode="rgb_array").shape
obs_space = (obs_space[2], obs_space[0], obs_space[1])
num_actions = env.action_space.n

model_free = ModelFree(obs_space, num_actions)

random_input = torch.rand((1, *obs_space))

writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, "model_free"))
writer.add_graph(model_free, random_input)
writer.close()
