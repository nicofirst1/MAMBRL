from itertools import chain

import torch
import torch.nn.functional as F
from rich.progress import track
from torch import optim
from torch.nn.utils import clip_grad_norm_

from src.common import parametrize_state, mas_dict2tensor, get_env_configs, Params
from src.env import get_env
from src.model import RolloutStorage, ModelFree, ImaginationCore, target_to_pix, I2A, EnvModel


def get_actor_critic(obs_space, params, num_rewards):
    """
    Create all the modules to build the i2a
    """

    num_colors = len(params.color_index)

    t2p = target_to_pix(params.color_index, gray_scale=params.gray_scale)

    env_model = EnvModel(
        obs_space,
        num_rewards=num_rewards,
        num_frames=params.num_frames,
        num_actions=5,
        num_colors=num_colors,
    )
    env_model = env_model.to(params.device)

    # fix: perchè passiamo il num_frames al model free ma poi non li usa
    # all'interno?
    model_free = ModelFree(obs_space, num_actions=5, )
    model_free = model_free.to(params.device)

    imagination = ImaginationCore(
        num_rollouts=1,
        in_shape=obs_space,
        num_actions=5,
        num_rewards=num_rewards,
        env_model=env_model,
        model_free=model_free,
        device=params.device,
        num_frames=params.num_frames,
        full_rollout=params.full_rollout,
        target2pix=t2p,
    )
    imagination = imagination.to(params.device)

    actor_critic = I2A(
        in_shape=obs_space,
        num_actions=5,
        num_rewards=num_rewards,
        hidden_size=256,
        imagination=imagination,
        full_rollout=params.full_rollout,
        num_frames=params.num_frames,
    )

    actor_critic = actor_critic.to(params.device)

    return actor_critic


def train(params: Params):
    env = get_env(get_env_configs(params))

    if params.resize:
        obs_space = params.obs_shape
    else:
        obs_space = env.render(mode="rgb_array").shape

    num_rewards = len(env.par_env.get_reward_range())
    ac_dict = {
        agent_id: get_actor_critic(obs_space, params, num_rewards)
        for agent_id in env.agents
    }

    optim_params = [list(ac.parameters()) for ac in ac_dict.values()]
    optim_params = chain.from_iterable(optim_params)

    optimizer = optim.RMSprop(
        optim_params, params.lr, eps=params.eps, alpha=params.alpha
    )

    rollout = RolloutStorage(
        params.horizon * params.episodes,
        obs_space,
        num_agents=params.agents,
        gamma=params.gamma,
        size_mini_batch=params.minibatch,
        num_actions=5,
    )
    rollout.to(params.device)

    obs_shape = (params.obs_shape[0] * params.num_frames, params.obs_shape[1], params.obs_shape[2])
    for ep in range(params.epochs):
        # fill rollout storage with trajcetories
        collect_trajectories(params, env, ac_dict, rollout, obs_shape)
        print('\n')
        # train for all the trajectories collected so far
        train_epoch(rollout, ac_dict, env, params, optimizer, optim_params, obs_shape)
        rollout.after_update()


# todo: this can be done in parallel
def collect_trajectories(params, env, ac_dict, rollout, obs_shape):
    """
    Collect a number of samples from the environment based on the current model (in eval mode)
    """
    state_fn = parametrize_state(params)
    state_channel = int(params.obs_shape[0])

    # set all i2a to eval
    [model.eval() for model in ac_dict.values()]

    for episode in track(range(params.episodes), description="Sample collection episode "):
        # init dicts and reset env
        dones = {agent_id: False for agent_id in env.agents}
        action_dict = {agent_id: False for agent_id in env.agents}
        values_dict = {agent_id: False for agent_id in env.agents}

        state = env.reset()
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
                action_logit, value_logit = ac_dict[agent_id](observation)

                # get action with softmax and multimodal (stochastic)
                action_log_probs = F.softmax(action_logit, dim=1)
                action = action_log_probs.multinomial(1).squeeze()
                action_dict[agent_id] = int(action)
                values_dict[agent_id] = int(value_logit)
                action_log_probs = torch.log(
                    action_log_probs).unsqueeze(dim=-1)
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
                action_log_probs=action_log_probs.detach().squeeze(),
            )

            # Update observation
            observation = observation.squeeze(dim=0)
            observation = torch.cat([observation[state_channel:, :, :], current_state], dim=0)


def train_epoch(rollouts, ac_dict, env, params, optimizer, optim_params, obs_shape):
    # todo: add logging in wandb

    # estimate advantages
    rollouts.compute_returns(rollouts.values[-1])
    advantages = rollouts.returns - rollouts.values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

    # get data generation that splits rollout in batches
    data_generator = rollouts.recurrent_generator(advantages, params.num_frames)
    num_batches = rollouts.get_num_batches(params.num_frames)

    # set model to train mode
    [model.train() for model in ac_dict.values()]

    # fix: commented for debug
    # for sample in track(data_generator, description="Batches", total=num_batches):
    for sample in data_generator:
        (
            states_batch,
            actions_batch,
            return_batch,
            masks_batch,
            old_action_log_probs_batch,
            adv_targ,
        ) = sample

        logits, action_log_probs, values, entropys = [], [], [], []

        for agent_id in env.agents:
            agent_index = env.agents.index(agent_id)
            agent_action = actions_batch[:, agent_index]

            agent = ac_dict[agent_id]
            logit, action_log_prob, value, entropy = agent.evaluate_actions(
                states_batch, agent_action
            )

            # add multi agent dim
            logits.append(logit.unsqueeze(dim=-1))
            action_log_probs.append(action_log_prob.unsqueeze(dim=-1))
            values.append(value.unsqueeze(dim=-1))
            entropys.append(entropy.unsqueeze(dim=-1))

        # unpack all the calls from before
        logits = torch.cat(logits, dim=-1)
        action_log_probs = torch.cat(action_log_probs, dim=-1)
        values = torch.cat(values, dim=-1)
        entropys = torch.cat(entropys, dim=-1)

        value_loss = (return_batch - values).pow(2).mean()

        # take last old_action_prob/adv_targ since is the prob of the last frame in the sequence of num_frames
        # old_action_log_probs_batch = old_action_log_probs_batch[:, -1]
        # adv_targ = adv_targ[:, -1:, :]
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

        adv_targ = adv_targ.view(adv_targ.shape[0], -1, adv_targ.shape[1])
        surr1 = ratio * adv_targ
        surr2 = (
                torch.clamp(
                    ratio,
                    1.0 - params.configs["ppo_clip_param"],
                    1.0 + params.configs["ppo_clip_param"],
                ) *
                adv_targ
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


if __name__ == '__main__':
    params = Params()
    train(params)