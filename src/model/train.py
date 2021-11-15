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

    optim_params = []
    for ac in ac_dict.values():
        optim_params += list(ac.parameters())

    optimizer = optim.RMSprop(optim_params, config['lr'], eps=config['eps'],
                              alpha=config['alpha'])

    state_fn = parametrize_state(params)

    rollout = RolloutStorage(params.num_steps, obs_space, num_agents=params.agents)
    if params.device == "cuda":
        rollout.cuda()

    all_rewards = []
    all_losses = []

    _ = env.reset()
    state = env.render(mode="rgb_array")
    current_state = state_fn(state)

    rollout.states[0].copy_(current_state)

    episode_rewards = torch.zeros(1, 1)
    final_rewards = torch.zeros(1, 1)

    """
        How to include the multi agent in this cicles?
    """
    for i in range(params.episodes):
        dones = {agent_id: False for agent_id in env.agents}
        action_dict = {agent_id: False for agent_id in env.agents}
        _ = env.reset()

        for step in range(params.num_steps):

            if current_state.ndim == 3:
                current_state = current_state.to(params.device).unsqueeze(dim=0)

            for agent_id in env.agents:
                if dones[agent_id]:
                    # skip action for done agents
                    action_dict[agent_id] = None
                    continue
                action = ac_dict[agent_id].act(current_state)
                action_dict[agent_id] = int(action)

            _, rewards, dones, _ = env.step(action_dict)

            ## Our reward is a dict {'agent_0': reward}
            episode_rewards += sum(rewards.values())

            ## Added by me

            if dones.pop("__all__"):
                # if done for all agents end episode
                break

            # sort in agent orders and convert to list of int for tensor

            masks = 1 - mas_dict2tensor(dones)
            rewards = mas_dict2tensor(rewards)

            next_state = env.render(mode="rgb_array")
            current_state = state_fn(next_state)

            # rollout.insert(step, current_state, action.data, reward, masks)
            rollout.insert(step, current_state, action, rewards, masks)

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

        logit, action_log_probs, values, entropy = tuple(map(torch.stack, zip(*tmp)))

        distil_logit, _, _, _ = distil_policy.evaluate_actions(
            rollout.states[:-1],
            rollout.actions
        )

        distil_loss = F.softmax(logit, dim=1).detach() * F.log_softmax(distil_logit, dim=1)
        distil_loss = distil_loss.sum(1).mean()
        distil_loss *= 0.01

        values = values.view(params.agents, params.num_steps, -1)

        action_log_probs = action_log_probs.view(params.agents, params.num_steps, -1)
        advantages = returns - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.data * action_log_probs).mean()

        distil_optimizer.zero_grad()
        distil_loss.backward()
        distil_optimizer.step()

        optimizer.zero_grad()
        loss = value_loss * config['value_loss_coef'] + action_loss - entropy * config['entropy_coef']
        loss = loss.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(optim_params, config['max_grad_norm'])
        optimizer.step()

        # wandb.log({"loss": loss.item()})
        # wandb.log({"reward": final_rewards.mean()})

        all_rewards.append(final_rewards.mean())
        all_losses.append(loss.item())
        print("Update {}:".format(i))
        print("\t Mean Loss: {}".format(np.mean(all_losses[-10:])))
        print("\t Last 10 Mean Reward: {}".format(np.mean(all_rewards[-10:])))
        # wandb.log({"Mean Reward": np.mean(all_rewards[-10:])})

        rollout.after_update()
