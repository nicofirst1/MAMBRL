import ray
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

from NavigationEnv import get_env
from NavigationModel import NavModel
from Parameters import Params


def get_env_configs(params: Params):
    ## ENV CONFIG
    env_config = dict(
        N=params.agents, landmarks=params.landmarks, max_cycles=25, continuous_actions=False,
        name=params.experiment_name
    )

    register_env(params.experiment_name, lambda config: get_env(config))

    return env_config


def get_policy_configs(params: Params):
    env = get_env(env_config)

    ModelCatalog.register_custom_model(params.model_name, NavModel)

    policies = {agnt: (None, env.observation_space, env.action_space, {}) for agnt in env.agents}
    policy_configs = dict(
        policies=policies,
        policy_mapping_fn=lambda agent_id: agent_id
    )

    return policy_configs


def get_model_configs(params: Params):
    model_configs = dict(custom_model=params.model_name,
                         vf_share_layers=True, )
    ModelCatalog.register_custom_model(params.model_name, NavModel)
    return model_configs


def get_general_configs(params: Params):
    configs = dict(
        env=params.experiment_name,
        num_gpus=params.num_gpus if not params.debug else 0,
        num_workers=params.num_workers if not params.debug else 0,
        framework=params.framework,
    )

    return configs


if __name__ == "__main__":
    params = Params()

    ray.init(local_mode=params.debug)

    # Get configs
    env_config = get_env_configs(params)
    policy_configs = get_policy_configs(params)
    model_configs = get_model_configs(params)
    configs = get_general_configs(params)

    # set specific configs in general dic
    configs['model'] = model_configs
    configs['env_config'] = env_config
    configs['multiagent'] = policy_configs

    ## TRAIING

    analysis = ray.tune.run(
        "PPO",
        name=f"{params.experiment_name}_test",
        metric="episode_reward_mean",
        config=configs)
    print(analysis)
