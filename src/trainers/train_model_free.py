"""nn_test file.

train the model free network
"""
import os
import sys
from itertools import chain
from typing import Tuple

import torch
import torch.optim as optim
import torch.nn.functional as F
from rich.progress import track

from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from src.env.NavEnv import get_env
from src.common.utils import get_env_configs, order_state
from src.common.Params import Params
from src.model.ModelFree import ModelFree
from src.model.RolloutStorage import RolloutStorage
from src.trainers.train_utils import mas_dict2tensor, collect_trajectories


def traj_collection_policy(ac_dict):
    def inner(
            agent_id: str, observation: torch.Tensor
    ) -> Tuple[int, int, torch.Tensor]:
        action_logit, value_logit = ac_dict[agent_id](observation)
        action_probs = F.softmax(action_logit, dim=1)
        action = action_probs.multinomial(1).squeeze()

        value = int(value_logit)
        action = int(action)

        return action, value, action_probs

    return inner


def train_epoch(rollout, ac_dict, env, optimizer, optim_params):
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

    # # get data generation that splits rollout in batches
    data_generator = rollout.recurrent_generator()
    num_mini_batches = rollout.get_num_batches()


    # MINI BATCHES ARE NECESSARY FOR PPO (SINCE IT USES OLD AND NEW POLICIES)
    for sample in data_generator:
        (
            states_mini_batch,
            actions_mini_batch,
            return_mini_batch,
            masks_mini_batch,
            old_action_log_probs_mini_batch,
            adv_targ_mini_batch,
            _
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

            agent = ac_dict[agent_id]
            _, action_log_prob, action_prob, value, entropy = \
                agent.evaluate_actions(
                    states_mini_batch,
                    actions_mini_batch[
                    :, agent_index].unsqueeze(-1)
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

        ratio = torch.exp(action_log_probs -
                          old_action_log_probs_mini_batch)

        surr1 = ratio * adv_targ_mini_batch
        surr2 = (
                torch.clamp(
                    ratio,
                    1.0 - params.ppo_clip_param,
                    1.0 + params.ppo_clip_param,

                ) *
                adv_targ_mini_batch
        )

        action_loss = -torch.min(surr1, surr2).mean()

        optimizer.zero_grad()
        loss = (
                value_loss * params.value_loss_coef +
                action_loss -
                entropys * params.entropy_coef
        )
        loss = loss.mean()
        loss.backward()

        clip_grad_norm_(optim_params,
                        params.max_grad_norm)
        optimizer.step()

    return states_mini_batch[0]


if __name__ == '__main__':

    TENSORBOARD_DIR = os.path.join(os.path.abspath(os.pardir), os.pardir,
                                   "tensorboard")
    params = Params()
    device = params.device
    # =============================================================================
    # ENV
    # =============================================================================
    env_config = get_env_configs(params)
    env = get_env(env_config)
    num_rewards = len(env.get_reward_range())
    num_agents = params.agents
    obs_shape = env.reset().shape
    # channels are inverted
    num_actions = env.action_spaces['agent_0'].n
    # =============================================================================
    # TRAINING PARAMS
    # =============================================================================
    size_minibatch = 4

    # PARAMETER SHARING, single network forall the agents. It requires an index to
    # recognize different agents
    PARAM_SHARING = False
    # Init a2c and rmsprop
    if not PARAM_SHARING:
        ac_dict = {
            agent_id: ModelFree(obs_shape, num_actions)
            for agent_id in env.agents
        }
        optim_params = [list(ac.parameters()) for ac in ac_dict.values()]
        optim_params = chain.from_iterable(optim_params)

        optimizer = optim.RMSprop(
            optim_params, params.lr, eps=params.eps, alpha=params.alpha
        )
    else:
        raise NotImplementedError

    rollout = RolloutStorage(params.horizon * params.episodes,

                             obs_shape,
                             num_agents=num_agents,
                             gamma=params.gamma,
                             size_mini_batch=params.minibatch,
                             num_actions=num_actions)
    rollout.to(device)

    for epoch in track(range(params.epochs)):
        [model.eval() for model in ac_dict.values()]

        collect_trajectories(
            params,
            env,
            rollout,
            obs_shape,
        )
        # set model to train mode
        [model.train() for model in ac_dict.values()]

        states_mini_batch=train_epoch(rollout, ac_dict, env, optimizer, optim_params)

    writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, "model_free_trained"))
    for agent_id in env.agents:
        agent_index = env.agents.index(agent_id)
        agent = ac_dict[agent_id]
        torch.save(agent.state_dict(), "ModelFree_agent_" + str(agent_index))

        writer.add_graph(agent, states_mini_batch.unsqueeze(dim=0))
    writer.close()
