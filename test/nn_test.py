import sys
import os
project_dir = os.path.abspath(os.path.join(os.path.curdir,
                                           os.path.pardir, os.path.pardir))
sys.path.insert(0, project_dir)
from src.model.ActorCritic import ActorCritic
