from random import randint

import torch
import torch.nn.functional as F
from rich.progress import track
from torch import optim, nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from src.common.Params import Params
from src.common.utils import get_env_configs, parametrize_state, mas_dict2tensor
from src.env.NavEnv import get_env
from src.model.EnvModel import EnvModel
from src.model.ImaginationCore import target_to_pix
# todo: this can be done in parallel
from src.model.RolloutStorage import RolloutStorage


def collect_random_trajectories(params, env, rollout, obs_shape):
    """
    Collect a number of samples from the environment based on the current model (in eval mode)
    """
    state_fn = parametrize_state(params)
    state_channel = int(params.obs_shape[0])

    for episode in track(range(params.episodes), description="Sample collection episode "):
        # init dicts and reset env
        dones = {agent_id: False for agent_id in env.agents}
        action_dict = {agent_id: False for agent_id in env.agents}

        state = env.reset()
        current_state = state_fn(state)

        # Insert first state
        rollout.states[episode * params.horizon] = current_state.unsqueeze(dim=0)

        ## Initialize Observation
        observation = torch.zeros(obs_shape)
        observation[-state_channel:, :, :] = current_state
        for step in range(params.horizon):
            observation = observation.to(params.device).unsqueeze(dim=0)

            # let every agent act
            for agent_id in env.agents:

                # skip action for done agents
                if dones[agent_id]:
                    action_dict[agent_id] = None
                    continue

                else:
                    action_dict[agent_id] = randint(0, params.num_actions - 1)

            # Our reward/dones are dicts {'agent_0': val0,'agent_1': val1}
            next_state, rewards, dones, _ = env.step(action_dict)

            # if done for all agents end episode
            if dones.pop("__all__"):
                break

            # sort in agent orders and convert to list of int for tensor
            masks = 1 - mas_dict2tensor(dones)
            rewards = mas_dict2tensor(rewards)
            actions = mas_dict2tensor(action_dict)

            current_state = state_fn(next_state)
            rollout.insert(
                step=step + episode * params.horizon,
                state=current_state,
                action=actions,
                values=torch.zeros(actions.shape),
                reward=rewards,
                mask=masks,
                action_log_probs=torch.zeros(actions.shape),
            )

            # Update observation
            observation = observation.squeeze(dim=0)
            observation = torch.cat([observation[state_channel:, :, :], current_state], dim=0)


def train_env_model(rollouts, env_model, target2pix, params, optimizer, obs_shape):
    # todo: add logging in wandb

    # estimate advantages
    rollouts.compute_returns(rollouts.values[-1])
    advantages = rollouts.returns - rollouts.values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    # get data generation that splits rollout in batches
    data_generator = rollouts.recurrent_generator(advantages, params.num_frames)
    num_batches = rollouts.get_num_batches(params.num_frames)

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

        reward_loss= (return_batch - imagined_reward).pow(2).mean()
        reward_loss = Variable(reward_loss, requires_grad=True)
        image_loss = criterion(imagined_state, output_states_batch)

        optimizer.zero_grad()
        loss = reward_loss + image_loss
        loss.backward()

        clip_grad_norm_(env_model.parameters(), params.configs["max_grad_norm"])
        optimizer.step()


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
        env_model.parameters(), params.configs["lr"], eps=params.configs["eps"], alpha=params.configs["alpha"]
    )

    env_model = env_model.to(params.device)
    env_model = env_model.train()
    collect_random_trajectories(params, env, rollout, obs_space)
    train_env_model(rollout, env_model, t2p, params, optimizer, obs_space)
