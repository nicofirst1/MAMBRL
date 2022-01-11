import random
import time

from rich import print

from src.common import Params, print_current_curriculum
from src.env import get_env, EnvWrapper
from src.model import PpoWrapper

params = Params()
# Get configs
env_configs = params.get_env_configs()

env_configs['horizon'] = 10
env_configs['visible'] = True
params.minibatch=2


env = EnvWrapper(
    env=get_env(env_configs),
    frame_shape=params.frame_shape,
    num_stacked_frames=params.num_frames,
    device=params.device
)

agent = PpoWrapper(env=env, config=params)



def test_learn():
    agent.guided_learning_prob=0

    for _ in range(4):
        agent.learn(3, full_log_prob=False)