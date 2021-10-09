import ray
from model.training import visual_train, tune_train
from utils.utils import *

if __name__ == "__main__":
    params = Params()
    ray.init(local_mode=params.debug)

    # Get configs
    env_config = get_env_configs(params)
    policy_configs = get_policy_configs(params)
    model_configs = get_model_configs(params)
    configs = get_general_configs(params)

    # set specific configs in general dic
    configs["model"] = model_configs
    configs["env_config"] = env_config
    configs["multiagent"] = policy_configs

    ## TRAIING
    visual_train(params, configs)
    #tune_train(params, configs)
