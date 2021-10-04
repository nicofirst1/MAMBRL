import os


class Params:

    #### TRAINING params

    num_workers = 6
    num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    framework = "tfe"
    debug = True

    #### ENV params
    agents = 2
    landmarks = 3
    horizon=100
    episodes=5
    env_name = "collab_nav"
    model_name = f"{env_name}_model"

    #### Config Dict
    configs={}
