from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from src.common import Params
from src.env.NavEnv import RawEnv
from src.model import RolloutStorage
from src.train.Policies import TrajCollectionPolicy, RandomAction


def mas_dict2tensor(agent_dict, type) -> torch.Tensor:
    """
    sort agent dict and convert to tensor of int

    Params
    ------
        agent_dict:
    """
    tensor = sorted(agent_dict.items())
    tensor = [type(elem[1]) for elem in tensor]
    return torch.as_tensor(tensor)

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
            action_log_probs = action_log_probs.unsqueeze(dim=0)

            observation = observation.squeeze(dim=0)
            rollout.insert(
                step=step + episode * params.horizon,
                state=observation,
                next_state=next_state,
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

        # Please Note: next step is needed to associate a value to the ending
        # state to compute the GAE (when the trajectory is < len_episode)
        # at the moment we don't need it since one trajectory is exactly
        # one episode
        # current_state = current_state.to(
        #     params.device).unsqueeze(dim=0)
        # for agent_id in env.agents:
        #     _, next_value, _ = policy.act(
        #         agent_id, current_state)
        #     values_dict[agent_id] = float(next_value)
        # next_value = mas_dict2tensor(values_dict)

    # estimate advantages
    rollout.compute_returns()


def train_epoch_PPO(
        rollout: RolloutStorage,
        ac_dict: Dict[str, nn.Module],
        env: RawEnv,
        optimizers: Dict[str, torch.optim.Optimizer],
        params: Params,
) -> Dict[str, List[int]]:
    """train_epoch_PPO method.

    Performs a PPO update on a full rollout storage (aka one epoch).
    The update also estimate the loss and does the backpropagation
    Parameters
    ----------
        rollout:
        ac_dict: Dictionary of agents, represented as modules
        env:
        optimizers: the optimizers
        params:

    Returns
    -------
        infos: dict of loss values for logging

    """
    # # get data generation that splits rollout in batches
    data_generator = rollout.recurrent_generator()
    infos = dict(
        value_loss=[],
        surrogate_loss=[],
        entropys=[],
        loss=[]
    )
    num_minibatches = rollout.get_num_minibatches()
    assert num_minibatches > 0, "Assertion Error: params.horizon*params.episodes should be greater than params.minibatches"

    # MINI BATCHES ARE NECESSARY FOR PPO (SINCE IT USES OLD AND NEW POLICIES)
    for sample in data_generator:
        (
            states_minibatch,
            actions_minibatch,
            return_minibatch,
            masks_minibatch,
            old_action_log_probs_minibatch,
            adv_targ_minibatch,
            _,
        ) = sample

        # states_minibatch = [minibatch_len, num_channels, width, height]
        # actions_minibatch = [minibatch_len, num_agents]
        # return_minibatch = [minibatch_len, num_agents]
        # masks_minibatch = [minibatch_len, num_agents]
        # old_action_log_probs_minibatch = [minibatch_len, num_agents, num_actions]
        # adv_targ_minibatch = [minibatch_len, num_agents]
        # next_states_minibatch = [minibatch_len, num_channels, width, height]

        for agent_id in env.agents:
            agent_index = env.agents.index(agent_id)
            agent = ac_dict[agent_id]
            #agent.train()

            actions, action_log_probs, action_probs, values, entropys = agent.compute_action_entropy(states_minibatch)

            loss, surrogate_loss, value_loss = compute_PPO_update(model=agent,
                optimizer=optimizers[agent_id], returns=return_minibatch[:, agent_index], values=values,
                action_log_probs=action_log_probs, old_action_log_probs=old_action_log_probs_minibatch[:, agent_index],
                adv_targs=adv_targ_minibatch[:, agent_index], entropys=entropys, params=params
            )

            #single_action_log_prob = action_log_probs.gather(-1, agent_actions)
            #single_action_old_log_prob = old_action_log_probs_minibatch[:, agent_index].gather(-1, agent_actions)

            #value_loss = (return_minibatch[:, agent_index] - values.squeeze(-1)).pow(2)

            #value_loss_mean = value_loss.mean()
            #ratio = torch.exp(single_action_log_prob -
            #                  single_action_old_log_prob)

            #ratio = ratio.squeeze()
            #ratio = ratio.nan_to_num(
            #    posinf=params.abs_max_loss, neginf=-params.abs_max_loss)
            # fix: this gives a lot of problem since ratio goes to -inf
            # and in the surrogate loss it's taken as min
            #surr1 = ratio * adv_targ_minibatch[:, agent_index]
            #surr2 = (
            #    torch.clamp(
            #        ratio,
            #        1.0 - params.ppo_clip_param,
            #        1.0 + params.ppo_clip_param,
            #    )
            #    * adv_targ_minibatch[:, agent_index]
            #)

            #surrogate_loss = -torch.min(surr1, surr2)
            #surrogate_loss_mean = surrogate_loss.mean()

            #optimizers[agent_id].zero_grad()
            #loss = (
            #    value_loss * params.value_loss_coef
            #    + surrogate_loss
            #    - entropys * params.entropy_coef
            #)
            #loss = loss.mean()
            #loss.backward()

            #clip_grad_norm_(model.parameters(), params.max_grad_norm)
            #optimizer.step()

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
    loss.backward(retain_graph=True)

    #clip_grad_norm_(model.parameters(), params.max_grad_norm)
    optimizer.step()

    return loss, value_loss, surrogate_loss
