from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from src.common import Params
from src.env.NavEnv import RawEnv
from src.model import RolloutStorage
from src.train.Policies import TrajCollectionPolicy, RandomAction


def mas_dict2tensor(agent_dict) -> torch.Tensor:
    """mas_dict2tensor function.

    sort agent dict and convert to tensor of int

    Params
    ------
        agent_dict:
    """
    tensor = sorted(agent_dict.items())
    tensor = [int(elem[1]) for elem in tensor]
    return torch.as_tensor(tensor)


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
        optimizer: the optimizers
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
    assert num_minibatches > 0, "Assertion Error: params.horizon*params.episodes " + \
        "should be greater than params.minibatches"

    # MINI BATCHES ARE NECESSARY FOR PPO (SINCE IT USES OLD AND NEW POLICIES)
    for sample in data_generator:
        (
            states_mini_batch,
            actions_mini_batch,
            return_mini_batch,
            masks_mini_batch,
            old_action_log_probs_mini_batch,
            adv_targ_mini_batch,
            _,
        ) = sample

        # states_mini_batch = [mini_batch_len, num_channels, width, height]
        # actions_mini_batch = [mini_batch_len, num_agents]
        # return_mini_batch = [mini_batch_len, num_agents]
        # masks_mini_batch = [mini_batch_len, num_agents]
        # old_action_log_probs_mini_batch = [mini_batch_len, num_agents,
        # num_actions]
        # adv_targ_mini_batch = [mini_batch_len, num_agents]
        # next_states_mini_batch = [mini_batch_len, num_channels, width,
        # height]

        for agent_id in env.agents:
            agent_index = env.agents.index(agent_id)
            agent = ac_dict[agent_id]
            agent_actions = actions_mini_batch[:,
                                               agent_index].unsqueeze(dim=-1)

            _, action_log_probs, action_prob, values, entropys = \
                agent.compute_action_entropy(
                    states_mini_batch
                )

            # we set the model in training mode after taking the action_probs
            # othewise it modifies them
            agent.train()
            single_action_log_prob = action_log_probs.gather(-1, agent_actions)
            single_action_old_log_prob = \
                old_action_log_probs_mini_batch[:, agent_index].gather(
                    -1, agent_actions)

            value_loss = (
                return_mini_batch[:, agent_index] - values.squeeze(-1)).pow(2)

            value_loss_mean = value_loss.mean()
            ratio = torch.exp(single_action_log_prob -
                              single_action_old_log_prob)

            ratio = ratio.squeeze()
            ratio = ratio.nan_to_num(
                posinf=params.abs_max_loss, neginf=-params.abs_max_loss)
            # fix: this gives a lot of problem since ratio goes to -inf
            # and in the surrogate loss it's taken as min
            surr1 = ratio * adv_targ_mini_batch[:, agent_index]
            surr2 = (
                torch.clamp(
                    ratio,
                    1.0 - params.ppo_clip_param,
                    1.0 + params.ppo_clip_param,
                )
                * adv_targ_mini_batch[:, agent_index]
            )

            surrogate_loss = -torch.min(surr1, surr2)
            surrogate_loss_mean = surrogate_loss.mean()

            optimizers[agent_id].zero_grad()
            loss = (
                value_loss * params.value_loss_coef
                + surrogate_loss
                - entropys * params.entropy_coef
            )
            loss = loss.mean()
            loss.backward()

            infos['value_loss'].append(float(value_loss_mean))
            infos['surrogate_loss'].append(float(surrogate_loss_mean))
            infos['entropys'].append(float(entropys))
            infos['loss'].append(float(loss))

            clip_grad_norm_(agent.parameters(), params.max_grad_norm)
            optimizers[agent_id].step()

    return infos


# todo: this can be done in parallel
def collect_trajectories(
        params: Params,
        env: RawEnv,
        rollout: RolloutStorage,
        obs_shape,
        policy: Optional[TrajCollectionPolicy] = None,
):
    """collect_trajectories function.

    Collect a number of samples from the environment based on the current model
    (in eval mode)

    Parameters
    ----------
        params:
        env:
        rollout:
        obs_shape:
        policy: Subclass of TrajCollectionPolicy, define how the action should
        be computed

    """
    # Please Note: everything work until params.horizon is equal to the episode
    # lenght (i.e. done is true only after params.horizon steps).
    # otherwise we should use a list instead of a tensor with predefined
    # dimension or a counter
    counter = 0
    if policy is None:
        policy = RandomAction(params.num_actions, params.device)

    for episode in range(params.episodes):
        # init dicts and reset env
        dones = {agent_id: False for agent_id in env.agents}
        action_dict = {agent_id: False for agent_id in env.agents}
        values_dict = {agent_id: False for agent_id in env.agents}
        action_log_dict = {agent_id: False for agent_id in env.agents}

        current_state = env.reset()

        for step in range(params.horizon):
            current_state = current_state.unsqueeze(dim=0)

            # let every agent act
            for agent_id in env.agents:

                # skip action for done agents
                if dones[agent_id]:
                    action_dict[agent_id] = None
                    continue

                # call forward method
                action, value, action_log_probs = policy.act(
                    agent_id, current_state)

                # get action with softmax and multimodal (stochastic)

                action_dict[agent_id] = action
                values_dict[agent_id] = value

                action_log_dict[agent_id] = action_log_probs

            # Our reward/dones are dicts {'agent_0': val0,'agent_1': val1}
            next_state, rewards, dones, infos = env.step(action_dict)

            # sort in agent orders and convert to list of int for tensor
            masks = {agent_done: dones[agent_done]
                     for agent_done in dones if agent_done != '__all__'}
            # mask is 1 if the agent can act, 0 otherwise
            masks = 1 - mas_dict2tensor(masks)
            rewards = mas_dict2tensor(rewards)
            actions = mas_dict2tensor(action_dict)
            values = mas_dict2tensor(values_dict)
            action_log_probs_list = [elem.unsqueeze(dim=0)
                                     for _, elem in action_log_dict.items()]
            action_log_probs = torch.cat(action_log_probs_list, 0)

            current_state = current_state.squeeze(0)
            rollout.insert(
                step=counter,
                state=current_state,
                next_state=next_state,
                action=actions,
                values=values,
                reward=rewards,
                mask=masks,
                action_log_probs=action_log_probs,
            )

            current_state = next_state.to(params.device)
            counter += 1
            # if done for all agents end episode
            if dones["__all__"]:
                break

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
