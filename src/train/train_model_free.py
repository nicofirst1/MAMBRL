"""nn_test file.

train the model free network
"""
import os
from itertools import chain

import torch
import torch.optim as optim
from rich.progress import track
from torch.utils.tensorboard import SummaryWriter

from logging_callbacks import PPOWandb
from src.common.Params import Params
from src.common.utils import get_env_configs
from src.env.NavEnv import get_env
from src.model.ModelFree import ModelFree, ModelFreeResnet
from src.model.RolloutStorage import RolloutStorage
from src.train.Policies import ExplorationMAS
from src.train.train_utils import (collect_trajectories, train_epoch_PPO)

if __name__ == "__main__":

    TENSORBOARD_DIR = os.path.join(
        os.path.abspath(os.pardir), os.pardir, "tensorboard")
    params = Params()
    params.device = torch.device("cpu")
    # =============================================================================
    # ENV
    # =============================================================================
    env_config = get_env_configs(params)
    env = get_env(env_config)
    obs_shape = params.obs_shape
    # channels are inverted
    num_actions = env.action_spaces["agent_0"].n

    ac_dict = {agent_id: ModelFreeResnet(in_shape=obs_shape, num_actions=num_actions).to(
        params.device) for agent_id in env.agents}
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
    rollout.to(params.device)

    # get logging step based on num of batches
    num_minibatches = rollout.get_num_minibatches()
    num_minibatches = int(num_minibatches * 0.01) + 1

    # wandb_callback = PPOWandb(
    #     train_log_step=num_batches,
    #     val_log_step=num_batches,
    #     project="model_free",
    #     opts={},
    #     models=ac_dict,
    #     horizon=params.horizon,
    #     mode="disabled" if params.debug else "online",
    # )

    # init policy
    policy = ExplorationMAS(ac_dict, params.num_actions)

    for epoch in range(params.epochs):  # track(range(params.epochs)):
        [model.eval() for model in ac_dict.values()]

        collect_trajectories(params, env, rollout, obs_shape, policy=policy)
        # set model to train mode
        [model.train() for model in ac_dict.values()]

        infos = train_epoch_PPO(
            rollout, ac_dict, env, optimizer, optim_params, params
        )

        infos['exploration_temp'] = [policy.epsilon]

       # wandb_callback.on_batch_end(infos,  epoch, rollout)
        policy.increase_temp(rollout.actions)
        # reset the rollout by overwriting it (saves memory)
        rollout.after_update()

    writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, "model_free_trained"))
    for agent_id in env.agents:
        agent_index = env.agents.index(agent_id)
        agent = ac_dict[agent_id]
        torch.save(agent.state_dict(), f"ModelFree_agent_{agent_index}.pt")

        #writer.add_graph(agent, states_mini_batch[0].unsqueeze(dim=0))
    writer.close()
