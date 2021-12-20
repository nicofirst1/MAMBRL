from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from src.common import Params
from src.env.NavEnv import RawEnv
from src.model import RolloutStorage
from src.train.Policies import TrajCollectionPolicy, RandomAction


def mas_dict2tensor(agent_dict, type) -> torch.Tensor:
    """
    sort agent dict and convert to tensor of type

    Params
    ------
        agent_dict:
        type:
    """

    tensor = sorted(agent_dict.items())
    if type is not None:
        tensor = [type(elem[1]) for elem in tensor]
    else:
        tensor = [elem[1] for elem in tensor]

    return torch.as_tensor(tensor)

# todo: this can be done in parallel
def collect_trajectories(
        params: Params,
        env: RawEnv,
        rollout: RolloutStorage,
        obs_shape,
        policy: Optional[TrajCollectionPolicy] = None,
):
    """collect_trajectories function.

    Collect a number of samples from the environment based on the current model (in eval mode)

    Parameters
    ----------
        params:
        env:
        rollout:
        obs_shape:
        policy: Subclass of TrajCollectionPolicy, define how the action should be computed
    """
    state_channel = int(obs_shape[0])

    if policy is None:
        policy = RandomAction(params.num_actions, params.device)

    for episode in range(params.episodes):
        # init dicts and reset env
        dones = {agent_id: False for agent_id in env.agents}
        action_dict = {agent_id: False for agent_id in env.agents}
        values_dict = {agent_id: False for agent_id in env.agents}
        action_log_dict = {agent_id: False for agent_id in env.agents}

        current_state = env.reset()

        # Insert first state
        # rollout.states[episode * params.horizon] = current_state.unsqueeze(dim=0)

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

                # call forward method
                action, value, action_log_probs = policy.act(agent_id, observation)

                # get action with softmax and multimodal (stochastic)

                action_dict[agent_id] = action
                values_dict[agent_id] = value
                action_log_dict[agent_id] = action_log_probs

            # Our reward/dones are dicts {'agent_0': val0,'agent_1': val1}
            next_state, rewards, dones, infos = env.step(action_dict)
            next_state = next_state.to(params.device)

            # sort in agent orders and convert to list of int for tensor
            masks = 1 - mas_dict2tensor(dones, int)
            rewards = mas_dict2tensor(rewards, int)
            actions = mas_dict2tensor(action_dict, int)
            values = mas_dict2tensor(values_dict, float)
            action_log_probs = mas_dict2tensor(action_log_dict, float)

            observation = observation.squeeze(dim=0)

            rollout.insert(
                step=step + episode * params.horizon,
                state=observation,
                next_state = next_state,
                action=actions,
                values=values,
                reward=rewards,
                mask=masks[:-1],
                action_log_probs=action_log_probs,
            )

            # Update observation
            observation = torch.cat(
                [observation[state_channel:, :, :], next_state], dim=0
            )

def train_epoch_PPO(
        rollout: RolloutStorage,
        ac_dict: Dict[str, nn.Module],
        env: RawEnv,
        optimizers: Dict[str, torch.optim.Optimizer],
        params: Params,
) -> Dict[str, List[int]]:
    """
    Performs a PPO update on a full rollout storage (aka one epoch).
    The update also estimate the loss and does the backpropagation
    Args:
        rollout:
        ac_dict: Dictionary of agents, represented as modules
        env:
        optimizers: The optimizer
        params:

    Returns:
        infos: dict of loss values for logging

    """

    rollout.compute_returns()
    rollout.to(params.device)

    # # get data generation that splits rollout in batches
    data_generator = rollout.recurrent_generator()
    infos = dict(
        value_loss=[],
        surrogate_loss=[],
        entropys=[],
        loss=[]
    )

    # MINI BATCHES ARE NECESSARY FOR PPO (SINCE IT USES OLD AND NEW POLICIES)
    for sample in data_generator:
        (
            states_minibatch,
            actions_minibatch,
            return_minibatch,
            masks_minibatch,
            old_action_logs_minibatch,
            adv_targ_minibatch,
            _,
        ) = sample

        # states_mini_batch = [mini_batch_len, num_channels, width, height]
        # actions_mini_batch = [mini_batch_len, num_agents]
        # return_mini_batch = [mini_batch_len, num_agents]
        # masks_mini_batch = [mini_batch_len, num_agents]
        # old_action_log_probs_mini_batch = [mini_batch_len, num_agents]
        # adv_targ_mini_batch = [mini_batch_len, num_agents]
        # next_states_mini_batch = [mini_batch_len, num_channels, width, height]

        for agent_id in env.agents:
            agent_index = env.agents.index(agent_id)
            agent_actions = actions_minibatch[:, agent_index]

            agent = ac_dict[agent_id]
            _, action_logs, action_probs, values, entropys = agent.evaluate_actions(states_minibatch, agent_actions)

            loss, value_loss, surrogate_loss = compute_PPO_update(agent, optimizers[agent_id], return_minibatch, values, action_logs,
                old_action_logs_minibatch, adv_targ_minibatch, entropys, params)

            infos['value_loss'].append(float(value_loss))
            infos['surrogate_loss'].append(float(surrogate_loss))
            infos['entropys'].append(float(entropys))
            infos['loss'].append(float(loss))

    return infos

def compute_PPO_update(
        model,
        optimizer: torch.optim.Optimizer,
        returns,
        values,
        action_log_probs,
        old_action_log_probs,
        adv_targs,
        entropys,
        params: Params
):
    """
    Compute a PPO update through backpropagation
    Args:
        model:
        optimizer:
        returns:
        values:
        action_log_probs:
        old_action_log_probs:
        adv_targs:
        entropys:
        params

    Return:
        loss:
        value_loss:
        action_loss:
    """
    value_loss = (returns - values).pow(2).mean()
    ratio = torch.exp(action_log_probs - old_action_log_probs)

    surr1 = ratio * adv_targs
    surr2 = (
            torch.clamp(
                ratio,
                1.0 - params.ppo_clip_param,
                1.0 + params.ppo_clip_param,
            )
            * adv_targs
    )

    surrogate_loss = -torch.min(surr1, surr2).mean()

    loss = (
            value_loss * params.value_loss_coef
            + surrogate_loss
            - entropys * params.entropy_coef
    ).mean()

    optimizer.zero_grad()
    loss.backward()

    clip_grad_norm_(model.parameters(), params.max_grad_norm)
    optimizer.step()

    return loss, value_loss, surrogate_loss
