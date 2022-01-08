"""nn_test file.

train the model free network
"""
import os
from itertools import chain

import torch
import torch.optim as optim
from rich.progress import track

from logging_callbacks import PPOWandb
from src.common.Params import Params
from src.common.utils import get_env_configs
from src.env.NavEnv import get_env
from src.model.ModelFree import ModelFreeResnet
from src.model.RolloutStorage import RolloutStorage
from src.train.Policies import EpsilonGreedy
from src.train.train_utils import (collect_trajectories, train_epoch_PPO)


def get_model_free(agents, restore, device):
    ac_dict = {agent_id: ModelFreeResnet(in_shape=obs_shape, num_actions=num_actions) for agent_id in
               agents}

    if restore:
        for agent_id, agent in ac_dict.items():
            agent.load_state_dict(torch.load(f"ModelFree_{agent_id}.pt"))

    ac_dict = {k: v.to(device) for k, v in ac_dict.items()}

    return ac_dict


if __name__ == "__main__":

    TENSORBOARD_DIR = os.path.join(os.path.abspath(os.pardir), os.pardir, "tensorboard")
    params = Params()
    device = params.device
    # =============================================================================
    # ENV
    # =============================================================================
    env_config = get_env_configs(params)
    env = get_env(env_config)
    obs_shape = env.reset().shape
    # channels are inverted
    num_actions = env.action_spaces["agent_0"].n
    ac_dict = get_model_free(env.agents, params.restore, device)

    optim_params = [list(ac.parameters()) for ac in ac_dict.values()]
    optim_params = chain.from_iterable(optim_params)
    optim_params = list(optim_params)

    optimizer = optim.RMSprop(
        optim_params, params.lr, eps=params.eps, alpha=params.alpha
    )

    rollout = RolloutStorage(
        params.horizon * params.episodes,
        obs_shape,
        num_agents=params.agents,
        gamma=params.gamma,
        size_minibatch=params.minibatch,
    )
    rollout.to(device)

    # get logging step based on num of batches
    num_batches = rollout.get_num_minibatches()
    num_batches = int(num_batches * 0.01) + 1

    wandb_callback = PPOWandb(
        train_log_step=num_batches,
        val_log_step=num_batches,
        project="model_free",
        opts={},
        models=ac_dict,
        horizon=params.horizon,
        mode="disabled" if params.debug else "online",
    )

    # init policy
    policy = EpsilonGreedy(ac_dict, params.num_actions)

    for epoch in track(range(params.epochs)):
        [model.eval() for model in ac_dict.values()]

        collect_trajectories(params, env, rollout, obs_shape, policy=policy)
        # set model to train mode
        [model.train() for model in ac_dict.values()]

        infos = train_epoch_PPO(
            rollout, ac_dict, env, optimizer, optim_params, params
        )

        infos['exploration_temp'] = [policy.epsilon]

        wandb_callback.on_batch_end(infos, epoch, rollout)
        policy.increase_temp(rollout.actions)
        rollout.after_update()

        if epoch % params.checkpoint_freq == 0:
            for agent_id, agent in ac_dict.items():
                torch.save(agent.state_dict(), f"ModelFree_{agent_id}.pt")

    # writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, "model_free_trained"))
    # writer.add_graph(agent, states_mini_batch[0].unsqueeze(dim=0))
    # writer.close()