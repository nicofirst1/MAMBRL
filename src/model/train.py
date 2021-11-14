import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from common import Params
from env.NavEnv import get_env
from model.ActorCritic import ActorCritic
from model.EnvModel import EnvModel
from model.I2A import I2A
from model.ImaginationCore import ImaginationCore
from model.RolloutStorage import RolloutStorage

from skimage.transform import resize

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(params: Params, config: dict):
    env = get_env(config['env_config'])

    #wandb.init(project="mbrl", dir=".", tags=["I2A"])
    #wandb.config.update(vars(args))

    if params.resize:
        obs_space = params.obs_shape
    else:
        obs_space = env.render(mode="rgb_array").shape

    act_space = env.action_space

    env_model = EnvModel(obs_space, obs_space[1]*obs_space[2], 1)

    distil_policy = ActorCritic(obs_space, 5)
    distil_optimizer = optim.Adam(distil_policy.parameters())

    imagination = ImaginationCore(1, obs_space, 5, 1, env_model, distil_policy, full_rollout=config['full_rollout'])
    actor_critic = I2A(obs_space, 5, 1, 256, imagination, full_rollout=config['full_rollout'])
    optimizer = optim.RMSprop(actor_critic.parameters(), config['lr'], eps=config['eps'], alpha=config['alpha'])

    env_model = env_model.to(device)
    distil_policy = distil_policy.to(device)
    actor_critic = actor_critic.to(device)

    rollout = RolloutStorage(config['num_steps'], 1, obs_space)
    if device == "cuda":
        rollout.cuda()

    all_rewards = []
    all_losses = []

    _ = env.reset()
    state = env.render(mode="rgb_array")
    state = np.moveaxis(state, -1, 0)
    if params.resize:
        state = resize(state, params.obs_shape)

    state = np.expand_dims(state, axis=0)
    current_state = torch.FloatTensor(np.float32(state))

    rollout.states[0].copy_(current_state)

    episode_rewards = torch.zeros(1, 1)
    final_rewards = torch.zeros(1, 1)

    """
        How to include the multi agent in this cicles?
    """
    for i in range(config['num_frames']):
        for step in range(config['num_steps']):
            for agent_id in env.agents:
                current_state = current_state.to(device)
                action = actor_critic.act(current_state)

                ###Addeded these lines to make it work
                action = action.cpu().numpy()[0]
                _, reward, done, _ = env.step({agent_id: action})
                #next_state, reward, done, _ = env.step(action.detach().cpu().numpy())

                ## Our reward is a dict {'agent_0': reward}
                ##reward = torch.FloatTensor(reward).unsqueeze(1)
                reward = reward[agent_id]
                episode_rewards += reward

                ## Added by me
                done = [int(x) for x in done.values()]
                done = done[:-1]

                masks = torch.FloatTensor(1 - np.array(done)).unsqueeze(1)
                final_rewards *= masks
                final_rewards += (1 - masks) * episode_rewards
                episode_rewards *= masks

                next_state = env.render(mode="rgb_array")
                next_state = np.moveaxis(next_state, -1, 0)
                if params.resize:
                    next_state = resize(next_state, params.obs_shape)
                next_state = np.expand_dims(next_state, axis=0)

                current_state = torch.FloatTensor(np.float32(next_state))
                #rollout.insert(step, current_state, action.data, reward, masks)
                rollout.insert(step, current_state, torch.Tensor(np.reshape(action, (1, 1))), torch.Tensor(np.reshape(reward, (1, 1))), masks)

        _, next_value = actor_critic(rollout.states[-1])
        next_value = next_value.data

        returns = rollout.compute_returns(next_value, config['gamma'])

        logit, action_log_probs, values, entropy = actor_critic.evaluate_actions(
            rollout.states[:-1].view(-1, *obs_space),
            rollout.actions.view(-1, 1)
        )

        distil_logit, _, _, _ = distil_policy.evaluate_actions(
            rollout.states[:-1].view(-1, *obs_space),
            rollout.actions.view(-1, 1)
        )

        distil_loss = 0.01 * (F.softmax(logit, dim=1).detach() * F.log_softmax(distil_logit, dim=1)).sum(1).mean()

        values = values.view(config['num_steps'], 1, 1)

        action_log_probs = action_log_probs.view(config['num_steps'], 1, 1)
        advantages = returns - values

        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.data * action_log_probs).mean()

        optimizer.zero_grad()
        loss = value_loss * config['value_loss_coef'] + action_loss - entropy * config['entropy_coef']
        loss.backward()
        nn.utils.clip_grad_norm_(actor_critic.parameters(), config['max_grad_norm'])
        optimizer.step()

        distil_optimizer.zero_grad()
        distil_loss.backward()
        optimizer.step()
        #wandb.log({"loss": loss.item()})
        #wandb.log({"reward": final_rewards.mean()})

        if i % 100 == 0:
            all_rewards.append(final_rewards.mean())
            all_losses.append(loss.item())
            print("Update {}:".format(i))
            print("\t Mean Loss: {}".format(np.mean(all_losses[-10:])))
            print("\t Last 10 Mean Reward: {}".format(np.mean(all_rewards[-10:])))
            #wandb.log({"Mean Reward": np.mean(all_rewards[-10:])})

        rollout.after_update()