import sys
from random import randint
from typing import Tuple

import cv2
import torch
from rich.progress import track

from src.common import Params
from src.env.NavEnv import RawEnv
from src.model import RolloutStorage


def mas_dict2tensor(agent_dict):
    # sort in agent orders and convert to list of int for tensor

    tensor = sorted(agent_dict.items())
    tensor = [int(elem[1]) for elem in tensor]
    return torch.as_tensor(tensor)


def parametrize_state(params):
    """
    Function used to resize image coming from env
    """

    def inner(state):

        if params.resize:
            state = state.squeeze()
            # PIL.Image.fromarray(state).show()

            state = cv2.resize(
                state,
                dsize=(params.obs_shape[2], params.obs_shape[1]),
                interpolation=cv2.INTER_AREA,
            )
            # PIL.Image.fromarray(state).show()

        # fixme: why are we not using long instead of floats?
        state = torch.LongTensor(state)
        # move channel on second dimension if present, else add 1
        if len(state.shape) == 3:
            state = state.permute(2, 0, 1)
        else:
            state = state.unsqueeze(dim=0)

        state = state.to(params.device)
        return state

    return inner


def random_action(
        agent_id: str, observation: torch.Tensor
) -> Tuple[int, int, torch.Tensor]:
    params = Params()

    action = randint(0, params.num_actions - 1)
    value = 0
    action_log = torch.ones((1, params.num_actions))
    action_log = action_log.to(params.device)

    return action, value, action_log


# todo: this can be done in parallel
def collect_trajectories(
        params: Params,
        env: RawEnv,
        rollout: RolloutStorage,
        obs_shape,
        policy_fn=random_action,
):
    """
    Collect a number of samples from the environment based on the current model (in eval mode)
    policy_fn: should be a function that gets an agent_id and an observation and returns
    action : int, value: int , action_log_probs : torch.Tensor
    """
    state_channel = int(obs_shape[0])

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
                action, value, action_probs = policy_fn(agent_id, observation)

                # get action with softmax and multimodal (stochastic)

                action_dict[agent_id] = action
                values_dict[agent_id] = value
                action_log_probs = torch.log(action_probs).squeeze(0)[int(action)]
                action_log_dict[agent_id] = 0e-10 \
                    if float(action_log_probs) == 0.0 else float(action_log_probs)

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

            current_state = next_state

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
