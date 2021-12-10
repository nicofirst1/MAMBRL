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
from src.trainers.train_utils import collect_trajectories


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
    optim_params = chain.from_iterable(optim_params)

    optimizer = optim.RMSprop(
        optim_params, params.lr, eps=params.eps, alpha=params.alpha
    )

    rollout = RolloutStorage(
        params.horizon * params.episodes,
        obs_shape,
        num_agents=params.agents,
        gamma=params.gamma,
        size_mini_batch=params.minibatch,
        num_actions=params.num_actions,
    )
    rollout.to(params.device)

    policy_fn = traj_collection_policy(ac_dict)


    for ep in track(range(params.epochs), description=f"Epochs"):
        # fill rollout storage with trajcetories
        collect_trajectories(params, env, rollout, obs_shape, policy_fn=policy_fn)
        # train for all the trajectories collected so far
        train_epoch(rollout, ac_dict, env, params, optimizer, optim_params, obs_shape)
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


def train_epoch(rollouts, ac_dict, env, params, optimizer, optim_params, obs_shape):
    # todo: add logging_callbacks in wandb

    # estimate advantages
    rollouts.compute_returns(rollouts.values[-1])
    # normalize advantages

    # get data generation that splits rollout in batches
    data_generator = rollouts.recurrent_generator()

    # set model to train mode
    [model.train() for model in ac_dict.values()]

    # fix: commented for debug
    # for sample in track(data_generator, description="Batches", total=num_batches):
    for sample in data_generator:
        (
            states_batch,
            actions_batch,
            return_batch,
            reward_batch,
            masks_batch,
            old_action_log_probs_batch,
            adv_targ,
        ) = sample

        logits, action_log_probs, values, entropys = [], [], [], []

        for agent_id in env.agents:
            agent_index = env.agents.index(agent_id)
            agent_action = actions_batch[:, agent_index]

            agent = ac_dict[agent_id]
            logit, action_log_prob, value, entropy = agent.evaluate_actions(
                states_batch, agent_action
            )

            # add multi agent dim
            logits.append(logit.unsqueeze(dim=-1))
            action_log_probs.append(action_log_prob.unsqueeze(dim=-1))
            values.append(value.unsqueeze(dim=-1))
            entropys.append(entropy.unsqueeze(dim=-1))

        # unpack all the calls from before
        logits = torch.cat(logits, dim=-1)
        action_log_probs = torch.cat(action_log_probs, dim=-1)
        values = torch.cat(values, dim=-1)
        entropys = torch.cat(entropys, dim=-1)

        value_loss = (return_batch - values).pow(2).mean()

        # take last old_action_prob/adv_targ since is the prob of the last frame in the sequence of num_frames
        # old_action_log_probs_batch = old_action_log_probs_batch[:, -1]
        # adv_targ = adv_targ[:, -1:, :]
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        adv_targ = adv_targ.view(adv_targ.shape[0], -1, adv_targ.shape[1])
        surr1 = ratio * adv_targ
        surr2 = (
            torch.clamp(
                ratio,
                1.0 - params.ppo_clip_param,
                1.0 + params.ppo_clip_param,
            )
            * adv_targ
        )
        action_loss = -torch.min(surr1, surr2).mean()

        optimizer.zero_grad()
        loss = (
            value_loss * params.value_loss_coef
            + action_loss
            - entropys * params.entropy_coef
        )
        loss = loss.mean()
        loss.backward()

        clip_grad_norm_(optim_params, params.max_grad_norm)
        optimizer.step()


if __name__ == "__main__":
    params = Params()
    train(params)
