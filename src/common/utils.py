import gym
import numpy as np
from ray.rllib.models import ModelCatalog

from src.env.NavEnv import get_env
from .Params import Params


def get_env_configs(params: Params):
    env_config = dict(
        N=params.agents,
        landmarks=params.landmarks,
        max_cycles=params.horizon,
        continuous_actions=False,
        name=params.env_name,
    )

    #register_env(params.env_name, lambda config: get_env(config))
    params.configs['env_config'] = env_config

    return env_config


def get_policy_configs(params: Params):
    env = get_env(params.configs['env_config'])
    #ModelCatalog.register_custom_model(params.model_name, NavModel)

    if params.obs_type == "image":
        shape = env.render(mode="rgb_array").shape
        obs_dim = gym.spaces.Box(
            low=0,
            high=255,
            shape=shape,
            dtype=np.uint8,
        )
    elif params.obs_type == "states":
        obs_dim = env.observation_space
    else:
        raise NotImplementedError(f"No observation space for name '{params.obs_type}'")

    policies = {
        agnt: (None, obs_dim, env.action_space, {}) for agnt in env.agents
    }
    policy_configs = dict(
        policies=policies,
        policy_mapping_fn=lambda agent_id: agent_id
    )

    params.configs['policy_configs'] = policy_configs
    return policy_configs


def get_model_configs(params: Params):
    model_configs = dict(
        custom_model=params.model_name,
        vf_share_layers=True,
    )
    #ModelCatalog.register_custom_model(params.model_name, NavModel)
    params.configs['model_configs'] = model_configs

    return model_configs


def get_general_configs(params: Params):
    configs = dict(
        env=params.env_name,
        eager=True,
        eager_tracing=False,
        num_gpus=params.num_gpus if not params.debug else 0,
        num_workers=params.num_workers if not params.debug else 0,
        batch_mode="complete_episode",
        train_batch_size=400,
        rollout_fragment_length=300,
        lr=3e-4,
        framework=params.framework,
    )

    return configs


def trial_name_creator(something):
    name = str(something).rsplit("_", 1)[0]
    name = f"{name}_{Params.unique_id}"
    return name
