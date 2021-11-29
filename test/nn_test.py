"""nn_test file.

train the model free network
"""
import sys
import os
import cv2
import torch
from torch import optim
import torch.nn.functional as F
project_dir = os.path.abspath(os.path.join(os.path.curdir,
                                           os.path.pardir))
sys.path.insert(0, project_dir)
from src.env.NavEnv import get_env
from src.common.utils import *
from src.model.ModelFree import ModelFree
from src.model.RolloutStorage import RolloutStorage


# =============================================================================
# UTILITIES
# =============================================================================
def order_state(state):
    assert len(state.shape) == 3, "State should have 3 dimensions"
    state = state.transpose((2, 0, 1))
    return torch.FloatTensor(state.copy())


def mas_dict2tensor(agent_dict):
    # sort in agent orders and convert to list of int for tensor

    tensor = sorted(agent_dict.items())
    tensor = [int(elem[1]) for elem in tensor]
    return torch.as_tensor(tensor)


color_index = [  # map index to RGB colors
    (0, 255, 0),  # green -> landmarks
    (0, 0, 255),  # blue -> agents
    (255, 255, 255),  # white -> background
]
params = Params()
params.resize = False
params.gray_scale = False
device = params.device
# =============================================================================
# ENV
# =============================================================================
env_config = get_env_configs(params)
env = get_env(env_config)
num_rewards = len(env.par_env.get_reward_range())
num_agents = params.agents
if params.resize:
    obs_space = params.obs_shape

else:
    obs_space = env.render(mode="rgb_array").shape
    # channels are inverted
    obs_space = (obs_space[2], obs_space[0], obs_space[1])
num_actions = env.action_space.n
# =============================================================================
# TRAINING PARAMS
# =============================================================================
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.5
size_minibatch = params.minibatch  # 2
epochs = params.epochs  # 1
steps_per_episode = params.horizon  # params.horizon
number_of_episodes = 10  # int(10e5)

# rmsprop hyperparams:
lr = 7e-4
eps = 1e-5
alpha = 0.99

# PARAMETER SHARING, single network forall the agents. It requires an index to
# recognize different agents
PARAM_SHARING = False
# Init a2c and rmsprop
if not PARAM_SHARING:
    ac_dict = {
        agent_id: ModelFree(obs_space, num_actions)
        for agent_id in env.agents
    }

    opt_dict = {
        agent_id: optim.RMSprop(ac_dict[agent_id].parameters(), lr, eps=eps,
                                alpha=alpha)
        for agent_id in env.agents
    }
else:
    raise NotImplementedError

rollout = RolloutStorage(steps_per_episode,
                         obs_space,
                         num_agents,
                         gamma,
                         size_minibatch,
                         num_actions)


# =============================================================================
# TRAIN
# =============================================================================
for epoch in range(epochs):
    # =============================================================================
    # COLLECT TRAJECTORIES
    # =============================================================================
    [model.eval() for model in ac_dict.values()]
    # just consider 1 episode, since we need to fix the rollout function to
    # accept more than one episode without overwriting
    # for _ in range(params.episodes):
    # init dicts and reset env
    dones = {agent_id: False for agent_id in env.agents}
    action_dict = {agent_id: False for agent_id in env.agents}
    values_dict = {agent_id: False for agent_id in env.agents}

    current_state = env.reset()
    current_state = order_state(current_state)

    for step in range(steps_per_episode):
        action_log_probs_list = []
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
            action_probs = F.softmax(action_logit, dim=1)
            action = action_probs.multinomial(1).squeeze()
            action_dict[agent_id] = int(action)
            values_dict[agent_id] = int(value_logit)
            action_log_probs = torch.log(
                action_probs).unsqueeze(dim=-1)
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

        current_state = next_state
        current_state = order_state(current_state)
        rollout.insert(
            step=step,
            state=current_state,
            action=actions,
            values=values,
            reward=rewards,
            mask=masks,
            action_log_probs=action_log_probs.detach().squeeze(),
        )
# =============================================================================
# TRAIN
# =============================================================================
    # estimate advantages
    rollout.compute_returns(rollout.values[-1])
    advantages = rollout.returns - rollout.values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    # # get data generation that splits rollout in batches
    data_generator = rollout.recurrent_generator(
        advantages, params.num_frames)
    num_batches = rollout.get_num_batches(params.num_frames)

    # set model to train mode
    [model.train() for model in ac_dict.values()]

    # MINI BATCHES ARE NECESSARY FOR PPO (SINCE IT USES OLD AND NEW POLICIES)
    for sample in data_generator:
        (
            states_batch,
            actions_batch,
            return_batch,
            masks_batch,
            old_action_log_probs_batch,
            adv_targ,
        ) = sample

        # logits, action_log_probs, values, entropys = [], [], [], []
        # states_batch = [batch, params.num_frames*num_channels, width, height]
        # actions_batch = [batch, params.num_frames, num_agents]
        # return_batch = [batch, params.num_frames, num_agents]
        # masks_batch = [batch, params.num_frames, num_agents]
        # old_action_log_probs_batch = [batch, params.num_frames, num_actions,
        # num_agents]
        # adv_targ = [batch, params.num_frames, num_agents]
        for mini_batch in range(len(states_batch)):
            for agent_id in env.agents:
                agent_index = env.agents.index(agent_id)

                agent = ac_dict[agent_id]
                # here we need a list of logit, action_log_prob etc. not just one

                # for state_batch in states_batch:
                #     for frame in range(0, len(state_batch), params.num_frames):
                #         single_state = state_batch[frame:params.num_frames]
                #         single_state = single_state.unsqueeze(dim=0)

                # fixed
                logit, action_log_prob, value, entropy = \
                    agent.evaluate_actions(
                        states_batch[mini_batch].unsqueeze(
                            dim=0).view(params.num_frames, *obs_space),
                        actions_batch[mini_batch, :, agent_index].view(-1, 1)
                    )

                # add multi agent dim
                # logits.append(logit.unsqueeze(dim=-1))
                # action_log_probs.append(action_log_prob.unsqueeze(dim=-1))
                # values.append(value.unsqueeze(dim=-1))
                # entropys.append(entropy.unsqueeze(dim=-1))

            # unpack all the calls from before
            # logits = torch.cat(logits, dim=-1)
            # action_log_probs = torch.cat(action_log_probs, dim=-1)
            # values = torch.cat(values, dim=-1)
            # entropys = torch.cat(entropys, dim=-1)

                value_loss = (return_batch[mini_batch, :, agent_index].view(-1, 1) -
                              value).pow(2).mean()

                # take last old_action_prob/adv_targ since is the prob of the last
                # frame in the sequence of num_frames
                # old_action_log_probs_batch = old_action_log_probs_batch[:, -1]
                # adv_targ = adv_targ[:, -1:, :]
                ratio = torch.exp(action_log_prob -
                                  old_action_log_probs_batch[mini_batch, :, :, agent_index])
                current_adv_targ = adv_targ[mini_batch,
                                            :, agent_index]  # .view(1, -1).repeat(5, 1)
                surr1 = torch.zeros(ratio.shape)
                surr2 = torch.zeros(ratio.shape)
                for i in range(len(current_adv_targ)):
                    surr1[i] = ratio[i] * current_adv_targ[i]
                    surr2[i] = (
                        torch.clamp(
                            ratio[i],
                            1.0 - params.configs["ppo_clip_param"],
                            1.0 + params.configs["ppo_clip_param"],

                        ) *
                        current_adv_targ[i]
                    )
                action_loss = -torch.min(surr1, surr2).mean()

                optimizer.zero_grad()
                loss = (
                    value_loss * params.configs["value_loss_coef"] +
                    action_loss -
                    entropys * params.configs["entropy_coef"]
                )
                loss = loss.mean()
                loss.backward()

                clip_grad_norm_(optim_params, params.configs["max_grad_norm"])
                optimizer.step()
for agent_id in env.agents:
    agent_index = env.agents.index(agent_id)

    agent = ac_dict[agent_id]
    torch.save(agent.state_dict(), "agent_" + str(agent_index))
