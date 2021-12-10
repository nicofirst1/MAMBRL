"""nn_test file.

train the model free network
"""
import os
import sys
from itertools import chain
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from rich.progress import track
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from logging_callbacks import PPOWandb
from src.common.Params import Params
from src.common.utils import get_env_configs, order_state
from src.env.NavEnv import get_env
from src.model.ModelFree import ModelFree
from src.model.RolloutStorage import RolloutStorage
from src.train.train_utils import (collect_trajectories, mas_dict2tensor,
                                   train_epoch_PPO)


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

    ac_dict = {agent_id: ModelFree(obs_shape, num_actions).to(params.device) for agent_id in env.agents}
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
        model_config=params.__dict__,
        out_dir=params.WANDB_DIR,
        opts={},
        mode="disabled" if params.debug else "online",
    )

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

        infos, states_mini_batch = train_epoch_PPO(
            rollout, ac_dict, env, optimizer, optim_params, params
        )

        wandb_callback.on_batch_end(infos, 0, epoch)
        rollout.after_update()

    writer = SummaryWriter(os.path.join(TENSORBOARD_DIR, "model_free_trained"))
    for agent_id in env.agents:
        agent_index = env.agents.index(agent_id)
        agent = ac_dict[agent_id]
        torch.save(agent.state_dict(), "ModelFree_agent_" + str(agent_index))

        writer.add_graph(agent, states_mini_batch.unsqueeze(dim=0))
    writer.close()
