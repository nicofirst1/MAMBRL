from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize
from torch import optim

from src.common import Params
from src.env.NavEnv import get_env
from src.model.ActorCritic import ActorCritic
from src.model.EnvModel import EnvModel
from src.model.I2A import I2A
from src.model.ImaginationCore import ImaginationCore
from src.model.RolloutStorage import RolloutStorage


def parametrize_state(params):
    """
    Function used to fix image coming from env
    """
    def inner(state):
        state = np.moveaxis(state, -1, 0)
        if params.resize:
            state = resize(state, params.obs_shape)

        state = torch.FloatTensor(state)
        return state

    return inner


def mas_dict2tensor(agent_dict):
    # sort in agent orders and convert to list of int for tensor

    tensor = sorted(agent_dict.items())
    tensor = [int(elem[1]) for elem in tensor]
    return torch.as_tensor(tensor)


def get_actor_critic(obs_space, distil_policy, params):
    """
    Create all the modules to build the i2a
    """
    env_model = EnvModel(obs_space, obs_space[1] * obs_space[2], 1)
    env_model = env_model.to(params.device)

    imagination = ImaginationCore(num_rolouts=1, in_shape=obs_space, num_actions=5, num_rewards=1, env_model=env_model,
                                  distil_policy=distil_policy, device=params.device,
                                  full_rollout=params.full_rollout)
    imagination = imagination.to(params.device)

    actor_critic = I2A(in_shape=obs_space, num_actions=5, num_rewards=1, hidden_size=256, imagination=imagination,
                       full_rollout=params.full_rollout)

    actor_critic = actor_critic.to(params.device)

    return actor_critic


def train(params: Params, config: dict):
    env = get_env(config['env_config'])

    # wandb.init(project="mbrl", dir=".", tags=["I2A"])
    # wandb.config.update(vars(args))

    if params.resize:
        obs_space = params.obs_shape
    else:
        obs_space = env.render(mode="rgb_array").shape

    distil_policy = ActorCritic(obs_space, num_actions=5)
    distil_optimizer = optim.Adam(distil_policy.parameters())
    distil_policy = distil_policy.to(params.device)

    ac_dict = {agent_id: get_actor_critic(obs_space, distil_policy, params) for agent_id in env.agents}

    ac_optim_params = [list(ac.parameters()) for ac in ac_dict.values()]
    ac_optim_params = chain.from_iterable(ac_optim_params)

    ac_optimizer = optim.RMSprop(ac_optim_params, config['lr'], eps=config['eps'],
                              alpha=config['alpha'])

    state_fn = parametrize_state(params)

    rollout = RolloutStorage(params.num_steps, obs_space, num_agents=params.agents)
    rollout.to(params.device)

    all_rewards = []
    all_losses = []

    episode_rewards = torch.zeros(1, 1)
    final_rewards = torch.zeros(1, 1)


    for i in range(params.episodes):
        # init dicts and reset env
        dones = {agent_id: False for agent_id in env.agents}
        action_dict = {agent_id: False for agent_id in env.agents}

        _ = env.reset()
        state = env.render(mode="rgb_array")
        current_state = state_fn(state)

        for step in range(params.num_steps):

            current_state = current_state.to(params.device).unsqueeze(dim=0)

            # let every agent act
            for agent_id in env.agents:

                # skip action for done agents
                if dones[agent_id]:
                    action_dict[agent_id] = None
                    continue

                action = ac_dict[agent_id].act(current_state)
                action_dict[agent_id] = int(action)

            ## Our reward/dones are dicts {'agent_0': val0,'agent_1': val1}
            next_state, rewards, dones, _ = env.step(action_dict)

            # todo: log rewards better
            episode_rewards += sum(rewards.values())

            # if done for all agents end episode
            if dones.pop("__all__"):
                break

            # sort in agent orders and convert to list of int for tensor
            masks = 1 - mas_dict2tensor(dones)
            rewards = mas_dict2tensor(rewards)
            actions = mas_dict2tensor(action_dict)

            current_state = state_fn(next_state)
            rollout.insert(step, current_state, actions, rewards, masks)

        # from now on the last dimension is the number of agents

        # get value associated with last state in rollout
        next_value = [ac_dict[id](rollout.states) for id in env.agents]
        next_value = [x[1] for x in next_value]
        next_value = torch.concat(next_value, dim=1)

        next_value = next_value.data

        returns = rollout.compute_returns(next_value, config['gamma'])


        tmp = [ac_dict[id].evaluate_actions(
            rollout.states[:-1],
            rollout.actions
        ) for id in env.agents]

        # unpack all the calls from before
        logit, action_log_probs, values, entropy = tuple(map(torch.stack, zip(*tmp)))

        distil_logit, _, _, _ = distil_policy.evaluate_actions(
            rollout.states[:-1],
            rollout.actions
        )

        # estiamte distil loss and backpropag
        distil_loss = F.softmax(logit, dim=1).detach() * F.log_softmax(distil_logit, dim=1)
        distil_loss = distil_loss.sum(1).mean()
        distil_loss *= 0.01

        distil_optimizer.zero_grad()
        distil_loss.backward()
        distil_optimizer.step()

        # estiamte other loss and backpropag
        values = values.view(params.agents, params.num_steps, -1)

        action_log_probs = action_log_probs.view(params.agents, params.num_steps, -1)
        advantages = returns - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.data * action_log_probs).mean()

        ac_optimizer.zero_grad()
        loss = value_loss * config['value_loss_coef'] + action_loss - entropy * config['entropy_coef']
        loss = loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(ac_optim_params, config['max_grad_norm'])
        ac_optimizer.step()



        all_rewards.append(final_rewards.mean())
        all_losses.append(loss.item())
        print("Update {}:".format(i))


