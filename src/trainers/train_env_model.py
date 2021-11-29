from random import randint
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from src.common import Params, get_env_configs
from src.env import get_env
from src.model import EnvModel, target_to_pix, RolloutStorage
from src.trainers.train_utils import collect_trajectories


def traj_collection_policy(agent_id: str, observation: torch.Tensor) -> Tuple[int, int, torch.Tensor]:
    action = randint(0, params.num_actions - 1)
    value = 0
    action_log_probs = torch.zeros((1, params.num_actions))

    return action, value, action_log_probs


def train_env_model(rollouts, env_model, target2pix, params, optimizer, obs_shape):
    # todo: add logging_callbacks in wandb

    # estimate advantages
    rollouts.compute_returns(rollouts.values[-1])
    advantages = rollouts.returns - rollouts.values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    # get data generation that splits rollout in batches
    data_generator = rollouts.recurrent_generator(advantages, params.num_frames)

    criterion = nn.MSELoss()

    # fix: commented for debug
    # for sample in track(data_generator, description="Batches", total=num_batches):
    for sample in data_generator:
        (
            states_batch,
            actions_batch,
            return_batch,
            _,
            _,
            _,
        ) = sample

        # discard last state since prediction is on the next one
        input_states_batch = states_batch[:-1]
        output_states_batch = states_batch[1:]
        actions_batch = actions_batch[:-1]
        return_batch = return_batch[:-1]

        onehot_action = torch.zeros(
            params.minibatch - 1, params.num_actions, *obs_shape[1:]
        )
        onehot_action[:, actions_batch] = 1
        inputs = torch.cat([input_states_batch, onehot_action], 1)
        inputs = inputs.to(params.device)

        # call forward method
        imagined_state, reward = env_model(inputs)

        # ========================================
        # from imagined state to real
        # ========================================
        # imagined outputs to real one
        imagined_state = F.softmax(imagined_state, dim=1).max(dim=1)[1]
        imagined_state = imagined_state.view(
            params.minibatch - 1
            , *obs_shape)
        imagined_state = target2pix(imagined_state)

        imagined_reward = F.softmax(reward, dim=1).max(dim=1)[1]

        reward_loss = (return_batch - imagined_reward).pow(2).mean()
        reward_loss = Variable(reward_loss, requires_grad=True)
        image_loss = criterion(imagined_state, output_states_batch)

        optimizer.zero_grad()
        loss = reward_loss + image_loss
        loss.backward()

        clip_grad_norm_(env_model.parameters(), params.configs["max_grad_norm"])
        optimizer.step()
        print(loss)


if __name__ == '__main__':

    params = Params()

    # ========================================
    # get all the configuration parameters
    # ========================================

    params.agents = 1
    env_config = get_env_configs(params)
    env = get_env(env_config)

    if params.resize:
        obs_space = params.obs_shape
    else:
        obs_space = env.render(mode="rgb_array").shape

    num_rewards = len(env.par_env.get_reward_range())

    num_colors = len(params.color_index)

    t2p = target_to_pix(params.color_index, gray_scale=params.gray_scale)

    rollout = RolloutStorage(
        params.horizon * params.episodes,
        obs_space,
        num_agents=params.agents,
        gamma=0.998,
        size_mini_batch=params.minibatch,
        num_actions=5,
    )
    rollout.to(params.device)

    # ========================================
    #  init the env model
    # ========================================

    env_model = EnvModel(
        obs_space,
        num_rewards=num_rewards,
        num_frames=params.num_frames,
        num_actions=5,
        num_colors=num_colors,
    )

    optimizer = optim.RMSprop(
        env_model.parameters(), params.lr, eps=params.eps, alpha=params.alpha
    )

    env_model = env_model.to(params.device)
    env_model = env_model.train()
    for ep in range(params.epochs):
        # fill rollout storage with trajcetories
        collect_trajectories(params, env, traj_collection_policy, rollout, obs_space)
        print('\n')
        # train for all the trajectories collected so far
        train_env_model(rollout, env_model, t2p, params, optimizer, obs_space)
        rollout.after_update()
        torch.save(env_model.state_dict(), "env_model.pt")
