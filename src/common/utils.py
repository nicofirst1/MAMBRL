import cv2
import gym
import numpy as np
import torch
from src.env import get_env
from src.common.Params import Params


def mas_dict2tensor(agent_dict):
    # sort in agent orders and convert to list of int for tensor

    tensor = sorted(agent_dict.items())
    tensor = [int(elem[1]) for elem in tensor]
    return torch.as_tensor(tensor)

def get_env_configs(params: Params):
    env_config = dict(
        max_cycles=params.horizon,
        continuous_actions=False,
        name=params.env_name,
        gray_scale=params.gray_scale,
        scenario_kwargs=dict(
            landmark_reward=1,
            max_landmark_counter=4,
            landmark_penalty=-2,
            num_agents=params.agents,
            num_landmarks=params.landmarks,
        ),
    )

    # register_env(params.env_name, lambda config: get_env(config))
    params.configs["env_config"] = env_config

    return env_config


def get_policy_configs(params: Params):
    env = get_env(params.configs["env_config"])
    # ModelCatalog.register_custom_model(params.model_name, NavModel)

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

    policies = {agnt: (None, obs_dim, env.action_space, {}) for agnt in env.agents}
    policy_configs = dict(
        policies=policies, policy_mapping_fn=lambda agent_id: agent_id
    )

    params.configs["policy_configs"] = policy_configs
    return policy_configs


def get_model_configs(params: Params):
    model_configs = dict(
        custom_model=params.model_name,
        vf_share_layers=True,
    )
    # ModelCatalog.register_custom_model(params.model_name, NavModel)
    params.configs["model_configs"] = model_configs

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


def rgb2gray(rgb, dimension):
    rgb = rgb.transpose(dimension, -1)
    const = torch.as_tensor([0.2989, 0.5870, 0.1140])

    rgb = rgb * const
    rgb = rgb.sum(dim=-1).unsqueeze(dim=-1)
    rgb = rgb.transpose(-1, dimension)

    return rgb


# just an hack for adding 3 frames, actually not needed
def parametrize_state(params):
    """
    Function used to fix image coming from env
    """

    def inner(state):
        if params.resize:
            state = state.squeeze()
            state = cv2.resize(
                state,
                dsize=(params.obs_shape[2], params.obs_shape[1]),
                interpolation=cv2.INTER_CUBIC,
            )

        # fixme: why are we not using long instead of floats?
        state = torch.FloatTensor(state).unsqueeze(dim=0)
        return state

    return inner
