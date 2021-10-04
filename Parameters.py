import os


class Params:

    #### TRAINING params

    num_workers = 6
    num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    framework = "tf"
    debug = False

    #### ENV params
    agents = 2
    landmarks = 3
    experiment_name = "collab_nav"
    model_name = f"{experiment_name}_model"
