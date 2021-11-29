from random import randint
from typing import Tuple

import torch
from rich.progress import track

from src.common import parametrize_state, mas_dict2tensor, Params


def random_action(agent_id: str, observation: torch.Tensor) -> Tuple[int, int, torch.Tensor]:

    action = randint(0, Params.num_actions - 1)
    value = 0
    action_log_probs = torch.zeros((1, Params.num_actions))
    action_log_probs= action_log_probs.to(Params.device)

    return action, value, action_log_probs


# todo: this can be done in parallel
def collect_trajectories(params, env, rollout, obs_shape, policy_fn=random_action, ):
    """
    Collect a number of samples from the environment based on the current model (in eval mode)
    policy_fn: should be a function that gets an agent_id and an observation and returns
    action : int, value: int , action_log_probs : torch.Tensor
    """
    state_fn = parametrize_state(params)
    state_channel = int(params.obs_shape[0])

    for episode in range(params.episodes):
        # init dicts and reset env
        dones = {agent_id: False for agent_id in env.agents}
        action_dict = {agent_id: False for agent_id in env.agents}
        values_dict = {agent_id: False for agent_id in env.agents}

        state = env.reset(mode="rgb_array")
        current_state = state_fn(state)

        # Insert first state
        rollout.states[episode * params.horizon] = current_state.unsqueeze(dim=0)

        ## Initialize Observation
        observation = torch.zeros(obs_shape)
        observation[-state_channel:, :, :] = current_state
        for step in range(params.horizon):
            action_log_probs_list = []
            observation = observation.to(params.device).unsqueeze(dim=0)

            # let every agent act
            for agent_id in env.agents:

                # skip action for done agents
                if dones[agent_id]:
                    action_dict[agent_id] = None
                    continue

                # call forward method
                action, value, action_probs = policy_fn(agent_id, observation)

                # get action with softmax and multimodal (stochastic)

                action_dict[agent_id] = action
                values_dict[agent_id] = value
                action_log_probs = action_probs.unsqueeze(dim=-1)
                action_log_probs_list.append(action_log_probs)

            # Our reward/dones are dicts {'agent_0': val0,'agent_1': val1}
            next_state, rewards, dones, _ = env.step(action_dict)

            # if done for all agents end episode
            if dones.pop("__all__"):
                break

            # sort in agent orders and convert to list of int for tensor
            masks = 1 - mas_dict2tensor(dones)
            rewards = mas_dict2tensor(rewards)
            actions = mas_dict2tensor(action_dict)
            values = mas_dict2tensor(values_dict)
            action_log_probs = torch.cat(action_log_probs_list, dim=-1)

            current_state = state_fn(next_state)
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
            observation = torch.cat([observation[state_channel:, :, :], current_state], dim=0)
