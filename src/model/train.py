from itertools import chain

import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import resize
from torch import optim
from torch.nn.utils import clip_grad_norm_

from src.common import Params
from src.env.NavEnv import get_env
from src.model.EnvModel import EnvModel
from src.model.I2A import I2A
from src.model.ImaginationCore import ImaginationCore
from src.model.ModelFree import ModelFree
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


def get_actor_critic(obs_space, params):
    """
    Create all the modules to build the i2a
    """
    env_model = EnvModel(obs_space, obs_space[1] * obs_space[2], 1)
    env_model = env_model.to(params.device)

    model_free = ModelFree(obs_space, num_actions=5)
    model_free = model_free.to(params.device)

    imagination = ImaginationCore(num_rolouts=1, in_shape=obs_space, num_actions=5, num_rewards=1, env_model=env_model,
                                  model_free=model_free, device=params.device,
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

    ac_dict = {agent_id: get_actor_critic(obs_space, params) for agent_id in env.agents}

    optim_params = [list(ac.parameters()) for ac in ac_dict.values()]
    optim_params = chain.from_iterable(optim_params)

    optimizer = optim.RMSprop(optim_params, config['lr'], eps=config['eps'],
                              alpha=config['alpha'])

    rollout = RolloutStorage(params.num_steps, obs_space, num_agents=params.agents, gamma=config['gamma'])
    rollout.to(params.device)

    # fill rollout storage with trajcetories
    collect_trajectories(params, env, ac_dict, rollout)
    train_epochs(rollout, ac_dict, env, params, optimizer, optim_params)


def collect_trajectories(params, env, ac_dict, rollout):
    """
    Collect a number of samples from the environment based on the current model (in eval mode)
    """
    state_fn = parametrize_state(params)

    # set all i2a to eval
    [model.eval() for model in ac_dict.values()]

    for i in range(params.episodes):
        # init dicts and reset env
        dones = {agent_id: False for agent_id in env.agents}
        action_dict = {agent_id: False for agent_id in env.agents}
        values_dict = {agent_id: False for agent_id in env.agents}

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

                # call forward method
                action_logit, value_logit = ac_dict[agent_id](current_state)

                # get action with softmax and multimodal (stochastic)
                action_probs = F.softmax(action_logit)
                actions = action_probs.multinomial(1)
                action_dict[agent_id] = int(actions)
                values_dict[agent_id] = int(value_logit)

            ## Our reward/dones are dicts {'agent_0': val0,'agent_1': val1}
            next_state, rewards, dones, _ = env.step(action_dict)

            # if done for all agents end episode
            if dones.pop("__all__"):
                break

            # sort in agent orders and convert to list of int for tensor
            masks = 1 - mas_dict2tensor(dones)
            rewards = mas_dict2tensor(rewards)
            actions = mas_dict2tensor(action_dict)
            values = mas_dict2tensor(values_dict)

            current_state = state_fn(next_state)
            rollout.insert(step, current_state, actions, values, rewards, masks,action_probs)


def train_epochs(rollouts, ac_dict, env, params, optimizer, optim_params):

    # estimate advantages
    # todo: check if correct from formula
    next_value = [ac_dict[id](rollouts.states) for id in env.agents]
    next_value = [x[1] for x in next_value]  # get values discard actions
    next_value = torch.concat(next_value, dim=1)

    rollouts.compute_returns(next_value[-1])
    advantages = rollouts.returns - rollouts.values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    # get data generation that splits rollout in batches
    data_generator = rollouts.recurrent_generator(advantages,
                                                  params.minibatch)

    for sample in data_generator:
        states_batch, actions_batch, \
        return_batch, masks_batch, old_action_log_probs_batch, \
        adv_targ = sample

        tmp = [ac_dict[id].evaluate_actions(
            states_batch,
            actions_batch
        ) for id in env.agents]

        # unpack all the calls from before
        logit, action_log_probs, values, entropy = tuple(map(torch.stack, zip(*tmp)))

        value_loss = (return_batch - values).pow(2).mean()

        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - params.config['ppo_clip_param'], 1.0 + params.config['ppo_clip_param']) * adv_targ
        action_loss=- torch.min(surr1, surr2).mean()

        optimizer.zero_grad()
        loss = value_loss * params.config['value_loss_coef'] + action_loss - entropy * params.config['entropy_coef']
        loss = loss.mean()
        loss.backward()

        clip_grad_norm_(optim_params, params.config['max_grad_norm'])
        optimizer.step()

    # distil_logit, _, _, _ = distil_policy.evaluate_actions(
    #     rollouts.states[:-1],
    #     rollouts.actions
    # )
    #
    # # estiamte distil loss and backpropag
    # distil_loss = F.softmax(logit, dim=1).detach() * F.log_softmax(distil_logit, dim=1)
    # distil_loss = distil_loss.sum(1).mean()
    # distil_loss *= 0.01
    #
    # distil_optimizer.zero_grad()
    # distil_loss.backward()
    # distil_optimizer.step()
    #
    # # estiamte other loss and backpropag
    #
    # action_log_probs = action_log_probs.view(params.agents, params.num_steps, -1)
    #
    # action_loss = -(advantages.data * action_log_probs).mean()
    #

    #
    #
    #
    # all_rewards.append(final_rewards.mean())
    # all_losses.append(loss.item())
    # print("Update {}:".format(i))
