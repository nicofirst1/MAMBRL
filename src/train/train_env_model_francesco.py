"""env_model_test file.

train the environment network
"""
import os
import sys

import torch
import torch.nn.functional as F
import tqdm
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from src.common.Params import Params
from src.common.utils import get_env_configs, mas_dict2tensor, order_state
from src.env.NavEnv import get_env
from src.model.EnvModel import EnvModel
from src.model.ImaginationCore import target_to_pix
from src.model.ModelFree import ModelFree
from src.model.RolloutStorage import RolloutStorage

TENSORBOARD_DIR = os.path.join(os.path.abspath(os.pardir), os.pardir, "tensorboard")

params = Params()
params.resize = False
params.gray_scale = False
device = params.device
color_index = params.color_index
num_colors = len(color_index)
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
size_minibatch = 4
epochs = 1  # 1
steps_per_episode = 21
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
    ac_dict = {agent_id: ModelFree(obs_space, num_actions) for agent_id in env.agents}
    model_dict = {
        agent_id: EnvModel(
            obs_space, num_rewards, params.num_frames, num_actions, num_colors
        )
        for agent_id in env.agents
    }
    opt_dict = {
        agent_id: optim.RMSprop(
            model_dict[agent_id].parameters(), lr, eps=eps, alpha=alpha
        )
        for agent_id in env.agents
    }
else:
    raise NotImplementedError

rollout = RolloutStorage(
    steps_per_episode, obs_space, num_agents, gamma, size_minibatch, num_actions
)
rollout.to(device)
# =============================================================================
# TRAIN
# =============================================================================
for epoch in tqdm(range(epochs)):
    # =============================================================================
    # COLLECT TRAJECTORIES
    # =============================================================================
    print(f"epoch {epoch}/{epochs}: collecting trajectories...")
    [model.eval() for model in ac_dict.values()]
    dones = {agent_id: False for agent_id in env.agents}
    action_dict = {agent_id: False for agent_id in env.agents}
    values_dict = {agent_id: False for agent_id in env.agents}
    action_log_dict = {agent_id: False for agent_id in env.agents}

    current_state = env.reset()
    current_state = order_state(current_state)
    rollout.states[0].copy_(current_state)

    for step in range(steps_per_episode):
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
            action_log_probs = torch.log(action_probs).squeeze(0)[int(action)]
            action_log_dict[agent_id] = (
                sys.float_info.min
                if float(action_log_probs) == 0.0
                else float(action_log_probs)
            )

        # Our reward/dones are dicts {'agent_0': val0,'agent_1': val1}
        next_state, rewards, dones, _ = env.step(action_dict)
        masks = {
            agent_done: dones[agent_done]
            for agent_done in dones
            if agent_done != "__all__"
        }
        # sort in agent orders and convert to list of int for tensor
        masks = 1 - mas_dict2tensor(masks)
        rewards = mas_dict2tensor(rewards)
        actions = mas_dict2tensor(action_dict)
        values = mas_dict2tensor(values_dict)
        action_log_probs = mas_dict2tensor(action_log_dict)

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
        # if done for all agents
        if dones["__all__"]:
            break
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

    # =============================================================================
    # TRAIN
    # =============================================================================

    # # get data generation that splits rollout in batches
    data_generator = rollout.recurrent_generator()
    num_mini_batches = rollout.get_num_batches()

    t2p = target_to_pix(params.color_index, gray_scale=params.gray_scale)
    criterion = nn.MSELoss()
    # MINI BATCHES ARE NECESSARY FOR PPO (SINCE IT USES OLD AND NEW POLICIES)
    print(f"epoch {epoch}/{epochs}: training the env_models...")
    for sample in data_generator:
        (
            states_mini_batch,
            actions_mini_batch,
            return_mini_batch,
            _,
            _,
            _,
            next_states_mini_batch,
        ) = sample
        # states_mini_batch = [mini_batch_len, num_channels, width, height]
        # actions_mini_batch = [mini_batch_len, num_agents]
        # return_mini_batch = [mini_batch_len, num_agents]
        # next_states_mini_batch = [mini_batch_len, num_channels, width,
        # height]
        # minibatches should be divided by agent in order to train their
        # personal model
        one_hot_actions_mini_batch = {agent_id: [] for agent_id in env.agents}
        for elem in range(len(states_mini_batch)):
            for agent_id in env.agents:
                one_hot_action = torch.zeros(1, num_actions, *obs_space[1:])
                agent_index = env.agents.index(agent_id)
                action_index = int(actions_mini_batch[elem, agent_index])
                one_hot_action[0, action_index] = 1
                one_hot_actions_mini_batch[agent_id].append(one_hot_action)

        # concatenate input and one_hot_actions
        for agent_id in env.agents:
            action_input = torch.cat(one_hot_actions_mini_batch[agent_id], 0)
            inputs = torch.cat([states_mini_batch, action_input], 1)
            inputs = inputs.to(params.device)

            # call forward method
            imagined_state, reward = model_dict[agent_id](inputs)

            # ========================================
            # from imagined state to real
            # ========================================
            # imagined outputs to real one
            imagined_state = F.softmax(imagined_state, dim=1).max(dim=1)[1]
            imagined_state = imagined_state.view(size_minibatch, *obs_space[1:])
            imagined_state = t2p(imagined_state)

            imagined_reward = F.softmax(reward, dim=1).max(dim=1)[1]

            reward_loss = (
                (return_mini_batch[:, agent_index] - imagined_reward).pow(2).mean()
            )
            reward_loss = Variable(reward_loss, requires_grad=True)
            image_loss = criterion(imagined_state, next_states_mini_batch)

            opt_dict[agent_id].zero_grad()
            loss = reward_loss + image_loss
            loss.backward()

            clip_grad_norm_(
                model_dict[agent_id].parameters(), params.configs["max_grad_norm"]
            )
            opt_dict[agent_id].step()

for agent_id in env.agents:
    agent_index = env.agents.index(agent_id)
    agent = model_dict[agent_id]
    torch.save(agent.state_dict(), "EnvModel_agent_" + str(agent_index))
