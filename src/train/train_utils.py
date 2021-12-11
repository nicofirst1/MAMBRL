from typing import Dict, List, Tuple, Optional

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from src.common import Params
from src.env.NavEnv import RawEnv
from src.model import RolloutStorage
from src.train.Policies import TrajCollectionPolicy, RandomAction


def mas_dict2tensor(agent_dict) -> torch.Tensor:
    """
    sort agent dict and convert to list of int of tensor

    Args:
        agent_dict:

    Returns:

    """

    tensor = sorted(agent_dict.items())
    tensor = [int(elem[1]) for elem in tensor]
    return torch.as_tensor(tensor)


def train_epoch_PPO(
        rollout: RolloutStorage,
        ac_dict: Dict[str, nn.Module],
        env: RawEnv,
        optimizer: torch.optim.Optimizer,
        optim_params: List[torch.nn.Parameter],
        params: Params,
) -> Tuple[Dict[str, List[int]], torch.Tensor]:
    """
    Performs a PPO update on a full rollout storage (aka one epoch).
    The update also estimate the loss and does the backpropagation
    Args:
        rollout:
        ac_dict: Dictionary of agents, represented as modules
        env:
        optimizer: The optimizer
        optim_params: A list of agent.parameters() chained together
        params:

    Returns:
        infos: dict of loss values for logging
        states_mini_batch: i dunno why this is here

    """

    # fixme: non ho capito questo cosa faccia
    """
        current_state = current_state.to(params.device).unsqueeze(dim=0)
        for agent_id in env.agents:
            _, next_value = ac_dict[agent_id](current_state)
            values_dict[agent_id] = float(next_value)
        next_value = mas_dict2tensor(values_dict)
        # estimate advantages
        rollout.compute_returns(next_value)
        advantages = rollout.gae
        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

   
    """

    rollout.compute_returns(rollout.values[-1])
    rollout.to(params.device)

    # # get data generation that splits rollout in batches
    data_generator = rollout.recurrent_generator()
    infos = dict(
        value_loss=[],
        action_loss=[],
        entropys=[],
        loss=[]
    )

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
        # old_action_log_probs_mini_batch = [mini_batch_len, num_agents]
        # adv_targ_mini_batch = [mini_batch_len, num_agents]
        # next_states_mini_batch = [mini_batch_len, num_channels, width,
        # height]

        action_probs, action_log_probs, values, entropys = [], [], [], []

        for agent_id in env.agents:
            agent_index = env.agents.index(agent_id)
            agent_action = actions_mini_batch[:, agent_index]

            agent = ac_dict[agent_id]
            _, action_log_prob, action_prob, value, entropy = agent.evaluate_actions(
                states_mini_batch, agent_action
            )
            action_probs.append(action_prob.unsqueeze(dim=-1))
            action_log_probs.append(action_log_prob.unsqueeze(dim=-1))
            values.append(value.unsqueeze(dim=-1))
            entropys.append(entropy.unsqueeze(dim=-1))

        action_probs = torch.cat(action_probs, dim=-1)
        action_log_probs = torch.cat(action_log_probs, dim=-1)
        values = torch.cat(values, dim=-1)
        entropys = torch.cat(entropys, dim=-1)

        value_loss = (return_mini_batch - values).pow(2).mean()

        ratio = torch.exp(action_log_probs - old_action_log_probs_mini_batch)

        surr1 = ratio * adv_targ_mini_batch
        surr2 = (
                torch.clamp(
                    ratio,
                    1.0 - params.ppo_clip_param,
                    1.0 + params.ppo_clip_param,
                )
                * adv_targ_mini_batch
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

        infos['value_loss'].append(float(value_loss))
        infos['action_loss'].append(float(action_loss))
        infos['entropys'].append(float(entropys))
        infos['loss'].append(float(loss))

        clip_grad_norm_(optim_params, params.max_grad_norm)
        optimizer.step()

    return infos


# todo: this can be done in parallel
def collect_trajectories(
        params: Params,
        env: RawEnv,
        rollout: RolloutStorage,
        obs_shape,
        policy: Optional[TrajCollectionPolicy] = None,
):
    """
    Collect a number of samples from the environment based on the current model (in eval mode)

    Args:
        params:
        env:
        rollout:
        obs_shape:
        policy: Subclass of TrajCollectionPolicy, define how the action should be computed

    Returns:

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

                # call forward method
                action, value, action_log_probs = policy.act(agent_id, observation)

                # get action with softmax and multimodal (stochastic)

                action_dict[agent_id] = action
                values_dict[agent_id] = value
                action_log_dict[agent_id] = (
                    0e-10 if action_log_probs.isinf() else float(action_log_probs)
                )

            # Our reward/dones are dicts {'agent_0': val0,'agent_1': val1}
            next_state, rewards, dones, infos = env.step(action_dict)

            # if done for all agents end episode
            if dones.pop("__all__"):
                break

            # sort in agent orders and convert to list of int for tensor
            masks = 1 - mas_dict2tensor(dones)
            rewards = mas_dict2tensor(rewards)
            actions = mas_dict2tensor(action_dict)
            values = mas_dict2tensor(values_dict)
            action_log_probs = mas_dict2tensor(action_log_dict)

            current_state = next_state.to(params.device)

            rollout.insert(
                step=step + episode * params.horizon,
                state=current_state,
                action=actions,
                values=values,
                reward=rewards,
                mask=masks,
                action_log_probs=action_log_probs.detach().squeeze(dim=0),
            )

            # Update observation
            observation = observation.squeeze(dim=0)
            observation = torch.cat(
                [observation[state_channel:, :, :], current_state], dim=0
            )
