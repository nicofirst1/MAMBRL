import os

import ray
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from NavigationEnv import get_env
from NavigationModel import NavModel

if __name__ == "__main__":
    debug = True
    agents = 2
    experiment_name = "collab_nav"
    model_name = f"{experiment_name}_model"

    ray.init(local_mode=True)

    ## ENV CONFIG
    env_config = dict(
        N=agents, landmarks=3, max_cycles=25, continuous_actions=False, name=experiment_name
    )

    register_env(experiment_name, lambda config: get_env(config))

    ## POLICY CONFIG
    env = get_env(env_config)

    model_configs = dict(custom_model=model_name,
                         vf_share_layers=True, )
    ModelCatalog.register_custom_model(model_name, NavModel)

    policies = {agnt: (None, env.observation_space, env.action_space, {}) for agnt in env.agents}
    multiagent_config = dict(
        policies=policies,
        policy_mapping_fn=lambda agent_id: agent_id
    )

    ## MODEL CONFIG

    model_configs = dict(custom_model=model_name,
                         vf_share_layers=True, )
    ModelCatalog.register_custom_model(model_name, NavModel)

    config = {"env": experiment_name, "simple_optimizer": True, "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
              "num_workers": 1 if not debug else 0, "framework": "tf", 'model': model_configs,
              'env_config': env_config, "multiagent": multiagent_config, }

    ## TRAIING

    analysis = ray.tune.run(
        "PPO",
        name=f"{experiment_name}_test",
        metric="episode_reward_mean",
        config=config)
    print(analysis)
