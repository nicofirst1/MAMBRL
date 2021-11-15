"""env_test file.

create and render a random environment
"""
import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.curdir,
                                           os.path.pardir, os.path.pardir))
sys.path.insert(0,project_dir)
from src.common.utils import *
from src.env.NavEnv import get_env

params = Params()
# Get configs
env_config = get_env_configs(params)
env = get_env(env_config)

