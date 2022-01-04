import torch

from src.common.Params import Params

def get_env_configs(params: Params):
    env_config = dict(
        horizon=params.horizon,
        continuous_actions=False,
        name=params.env_name,
        gray_scale=params.gray_scale,
        obs_shape=params.obs_shape[2],
        num_actions=params.num_actions,
        visible=False,
        device=params.device,
        scenario_kwargs=dict(
            landmark_reward=1,
            max_landmark_counter=4,
            landmark_penalty=-2,
            num_agents=params.agents,
            num_landmarks=params.landmarks,
            max_size=3,
        ),
    )

    # register_env(params.env_name, lambda config: get_env(config))
    # params.configs["env_config"] = env_config
    return env_config


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

