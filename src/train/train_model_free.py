"""nn_test file.

train the model free network
"""
import os
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from src.env.NavEnv import get_env
from src.train.train_utils import mas_dict2tensor
from src.common.utils import get_env_configs, order_state
from src.common.Params import Params
from src.model.ModelFree import ModelFree
from src.model.RolloutStorage import RolloutStorage


TENSORBOARD_DIR = os.path.join(os.path.abspath(os.pardir), os.pardir,
                               "tensorboard")
params = Params()
params_dict = dict(Params.__dict__)
params.gray_scale = False
device = params.device
# =============================================================================
# ENV
# =============================================================================
env_config = get_env_configs(params)
env_config["scenario_kwargs"]["num_landmarks"] = 1
# fix: env_config non tiene in considerazione la modalità del rendering,
# aggiungere attributo
env = get_env(env_config)

num_rewards = len(env.get_reward_range())
num_agents = env.num_agents
if params.resize:
    # fix: Nicolò quando il params.resize è True ritorna solo un intero (32),
    # dovrebbe ritornare tutta la obs_shape (32, 32, 3) o (3, 32, 32)
    obs_space = (3, env.obs_shape, env.obs_shape)

else:
    # fix: Nicolò errore quando si fa il rendering:
    # AttributeError: 'RawEnv' object has no attribute 'render_geoms'
    obs_space = env.render(mode="rgb_array").shape
    # channels are inverted
    # obs_space = (obs_space[2], obs_space[0], obs_space[1])

# fix: Nicolò num_actions ed altri attributi dovrebbero essere parametri
# interni della classe non del file di configurazione (Params())
num_actions = params.num_actions
# =============================================================================
# TRAINING PARAMS
# =============================================================================
gamma = 0.99
entropy_coef = 0.01
value_loss_coef = 0.5
max_grad_norm = 0.5
size_minibatch = 4
epochs = 1
steps_per_episode = 21  # params.horizon
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
        agent_id: optim.RMSprop(
            ac_dict[agent_id].parameters(), lr, eps=eps, alpha=alpha)
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
    # just consider 1 episode, since we need to fix the rollout function to
    # accept more than one episode without overwriting
    # init dicts and reset env
    dones = {agent_id: False for agent_id in env.agents}
    action_dict = {agent_id: False for agent_id in env.agents}
    values_dict = {agent_id: False for agent_id in env.agents}
    action_log_dict = {agent_id: False for agent_id in env.agents}

    # fix: Nicolò env.reset() ritorna None
    current_state = env.reset()
    current_state = torch.randint(0, 255, (3, 32, 32)).float()
    # current_state = order_state(current_state)
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
            values_dict[agent_id] = float(value_logit)
            action_log_probs = torch.log(action_probs).squeeze(0)[int(action)]
            action_log_dict[agent_id] = sys.float_info.min \
                if float(action_log_probs) == 0.0 else float(action_log_probs)

        # Our reward/dones are dicts {'agent_0': val0,'agent_1': val1}
        next_state, rewards, dones, _ = env.step(action_dict)
        masks = {agent_done: dones[agent_done]
                 for agent_done in dones if agent_done != '__all__'}
        # sort in agent orders and convert to list of int for tensor
        masks = 1 - mas_dict2tensor(masks)
        rewards = mas_dict2tensor(rewards)
        actions = mas_dict2tensor(action_dict)
        values = mas_dict2tensor(values_dict)
        action_log_probs = mas_dict2tensor(action_log_dict)

        # fix: Nicolò l'environment torna tensori uint8, la rete mi accetta
        # solo i float, quindi va fatto il recast all'interno
        current_state = next_state.float()
        # current_state = order_state(current_state)

        rollout.insert(
            step=step,
            state=current_state,
            action=actions,
            values=values,
            reward=rewards,
            mask=masks,
            action_log_probs=action_log_probs,
        )
        # if done for all agents
        if dones["__all__"]:
            break

    current_state = current_state.to(params.device).unsqueeze(dim=0).float()
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
    num_minibatches = rollout.get_num_minibatches()

    # set model to train mode
    [model.train() for model in ac_dict.values()]

    print(f"epoch {epoch}/{epochs}: training the model_free...")
    # MINI BATCHES ARE NECESSARY FOR PPO (SINCE IT USES OLD AND NEW POLICIES)
    for sample in data_generator:
        (
            states_minibatch,
            actions_minibatch,
            return_minibatch,
            masks_minibatch,
            old_action_log_probs_minibatch,
            adv_targ_minibatch,
            _
        ) = sample
        # REMOVE
        states_minibatch = states_minibatch.float()
        for agent_id in env.agents:
            agent_index = env.agents.index(agent_id)

            agent = ac_dict[agent_id]
            _, action_log_prob, probs, value, entropy = \
                agent.evaluate_actions(
                    states_minibatch,
                    actions_minibatch[
                        :, agent_index].unsqueeze(-1)
                )

            value_loss = (return_minibatch[:, agent_index] -
                          value).pow(2).mean()

            ratio = torch.exp(action_log_prob -
                              old_action_log_probs_minibatch)

            surr1 = ratio * adv_targ_minibatch
            # fix: i parametri in params sono da aggiustare un po, ad esempip
            # il ppo_clip_params è sia un attributo della classe ed è sia
            # ripetuto in params.config
            surr2 = (
                torch.clamp(
                    ratio,
                    1.0 - params.ppo_clip_param,
                    1.0 + params.ppo_clip_param,

                ) *
                adv_targ_minibatch
            )

            action_loss = -torch.min(surr1, surr2).mean()

            opt_dict[agent_id].zero_grad()
            loss = (
                value_loss * params.value_loss_coef +
                action_loss -
                entropy * params.entropy_coef
            )
            loss = loss.mean()
            loss.backward()

            clip_grad_norm_(agent.parameters(),
                            params.max_grad_norm)
            opt_dict[agent_id].step()

writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, "model_free_trained"))
for agent_id in env.agents:
    agent_index = env.agents.index(agent_id)
    agent = ac_dict[agent_id]
    torch.save(agent.state_dict(), "ModelFree_agent_" + str(agent_index))

    writer.add_graph(agent, states_minibatch[0].unsqueeze(dim=0))
writer.close()
