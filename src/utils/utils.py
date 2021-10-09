from .Params import Params
from env.NavEnv import get_env
from model.NavModel import NavModel
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env

def get_env_configs(params: Params):
    env_config = dict(
        N=params.agents,
        landmarks=params.landmarks,
        max_cycles=params.horizon,
        continuous_actions=False,
        name=params.env_name,
    )

    register_env(params.env_name, lambda config: get_env(config))

    params.configs['env_config'] = env_config

    return env_config


def get_policy_configs(params: Params):
    env = get_env(params.configs['env_config'])

    ModelCatalog.register_custom_model(params.model_name, NavModel)

    policies = {
        agnt: (None, env.observation_space, env.action_space, {}) for agnt in env.agents
    }
    policy_configs = dict(
        policies=policies, policy_mapping_fn=lambda agent_id: agent_id
    )

    params.configs['policy_configs'] = policy_configs

    return policy_configs


def get_model_configs(params: Params):
    model_configs = dict(
        custom_model=params.model_name,
        vf_share_layers=True,
    )
    ModelCatalog.register_custom_model(params.model_name, NavModel)

    params.configs['model_configs'] = model_configs

    return model_configs


def get_general_configs(params: Params):
    configs = dict(
        env=params.env_name,
        num_gpus=params.num_gpus if not params.debug else 0,
        num_workers=params.num_cpus if not params.debug else 0,
        framework=params.framework,
    )

    return configs