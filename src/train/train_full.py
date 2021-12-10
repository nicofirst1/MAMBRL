from itertools import chain
from typing import Tuple

import torch
import torch.nn.functional as F
from rich.progress import track
from torch import optim
from torch.nn.utils import clip_grad_norm_

from src.common import Params, get_env_configs
from src.env import get_env
from src.model import (I2A, EnvModel, ImaginationCore, ModelFree,
                       RolloutStorage, target_to_pix)
from src.train.train_utils import collect_trajectories, train_epoch_PPO


def get_actor_critic(obs_space, params, reward_range):
    """
    Create all the modules to build the i2a
    """

    num_colors = len(params.color_index)
    num_rewards = len(reward_range)

    t2p = target_to_pix(params.color_index, gray_scale=params.gray_scale)

    env_model = EnvModel(
        obs_space,
        reward_range=reward_range,
        num_frames=params.num_frames,
        num_actions=params.num_actions,
        num_colors=num_colors,
        target2pix=t2p,
    )
    env_model = env_model.to(params.device)

    model_free = ModelFree(obs_space, num_actions=params.num_actions)
    model_free = model_free.to(params.device)

    imagination = ImaginationCore(
        num_rollouts=1,
        in_shape=obs_space,
        num_actions=params.num_actions,
        num_rewards=num_rewards,
        env_model=env_model,
        model_free=model_free,
        device=params.device,
        num_frames=params.num_frames,
        full_rollout=params.full_rollout,
        target2pix=t2p,
    )
    imagination = imagination.to(params.device)

    actor_critic = I2A(
        in_shape=obs_space,
        num_actions=params.num_actions,
        num_rewards=num_rewards,
        hidden_size=256,
        imagination=imagination,
        full_rollout=params.full_rollout,
        num_frames=params.num_frames,
    )

    actor_critic = actor_critic.to(params.device)

    return actor_critic


def train(params: Params):
    configs = get_env_configs(params)
    configs["mode"] = "rgb_array"
    env = get_env(configs)

    obs_shape = env.reset().shape

    reward_range = env.get_reward_range()
    ac_dict = {
        agent_id: get_actor_critic(obs_shape, params, reward_range)
        for agent_id in env.agents
    }

    optim_params = [list(ac.parameters()) for ac in ac_dict.values()]
    optim_params = list(chain.from_iterable(optim_params))

    optimizer = optim.RMSprop(
        optim_params, params.lr, eps=params.eps, alpha=params.alpha
    )

    rollout = RolloutStorage(
        params.horizon * params.episodes,
        obs_shape,
        num_agents=params.agents,
        gamma=params.gamma,
        size_minibatch=params.minibatch,
    )
    rollout.to(params.device)

    policy_fn = traj_collection_policy(ac_dict)

    for ep in track(range(params.epochs), description=f"Epochs"):
        # fill rollout storage with trajcetories
        collect_trajectories(params, env, rollout, obs_shape, policy_fn=policy_fn)
        # train for all the trajectories collected so far
        train_epoch_PPO(rollout, ac_dict, env, optimizer, optim_params, params)
        rollout.after_update()


def traj_collection_policy(ac_dict):
    def inner(
        agent_id: str, observation: torch.Tensor
    ) -> Tuple[int, int, torch.Tensor]:
        action_logit, value_logit = ac_dict[agent_id](observation)
        action_probs = F.softmax(action_logit, dim=1)
        action = action_probs.multinomial(1).squeeze()

        value = int(value_logit)
        action = int(action)

        return action, value, action_probs

    return inner


if __name__ == "__main__":
    params = Params()
    train(params)
