import os
import multiprocessing
import tensorflow as tf

class Params:
    debug = False

    #### TRAINING params
    num_cpus = multiprocessing.cpu_count() if not debug else 1
    num_gpus = len(tf.config.list_physical_devices('GPU')) if not debug else 0
    framework = "tf"

    #### ENV params
    agents = 2
    landmarks = 3
    horizon = 100
    episodes = 5
    env_name = "collab_nav"
    model_name = f"{env_name}_model"

    #### Config Dict
    configs={}
