from common.utils import *
from model.train import train

if __name__ == "__main__":
    params = Params()
    # ray.init(local_mode=params.debug)

    # Get configs
    env_config = get_env_configs(params)
    policy_configs = get_policy_configs(params)
    model_configs = get_model_configs(params)

    configs = {
        "env": params.env_name,
        "env_config": env_config,
        "framework": params.framework,
        "num_workers": params.num_workers,
        "num_gpus": params.num_gpus,
        "batch_mode": "complete_episodes",
        # "train_batch_size": 400,
        "rollout_fragment_length": 50,
        # PPO parameter
        "lr": 3e-4,
        "lambda": 0.95,
        "gamma": 0.998,
        "entropy_coef": 0.01,
        "clip_param": 0.2,
        "use_critic": True,
        "use_gae": True,
        "grad_clip": 5,
        "num_sgd_iter": 10,
        "value_loss_coef": 0.5,
        "max_grad_norm": 0.5,
        "eps": 1e-5,
        "alpha": 0.99,
        "ppo_clip_param": 0.1,
        # Callbacks
        "callbacks": {},
        # Model
        "model": model_configs,
        # Multiagent
        "multiagent": policy_configs,
    }

    params.configs = configs

    ## TRAINING
    train(params, configs)
