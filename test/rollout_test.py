"""rollout_test file.

analyze in real time the trajectories collected
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

from src.common.Params import Params
from src.common.utils import get_env_configs
from src.env.NavEnv import get_env
from src.train.train_utils import collect_trajectories
from src.model.RolloutStorage import RolloutStorage


if __name__ == "__main__":

    params = Params()
    params.horizon = 5
    params.device = torch.device("cpu")
    # =============================================================================
    # ENV
    # =============================================================================
    env_config = get_env_configs(params)
    env = get_env(env_config)
    obs_shape = params.obs_shape
    # channels are inverted
    num_actions = env.action_spaces["agent_0"].n

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

    collect_trajectories(params, env, rollout, obs_shape)
    rewards_np = np.array(rollout.rewards)
    states_np = np.array(rollout.states)
    masks_np = np.array(rollout.masks)
    actions_np = np.array(rollout.actions)
    action_log_probs_np = np.array(rollout.action_log_probs)
    values_np = np.array(rollout.values)
    returns_np = np.array(rollout.returns)
    gae_np = np.array(rollout.gae)

    # max 30 subplot otherwise it will fail the rendering
    n_samples = len(states_np)
    max_elem = min(n_samples, 30)
    fig = plt.figure(figsize=(14, 14))
    if max_elem == 30:
        columns = 6
        rows = 5
    else:
        columns = math.ceil(max_elem**(1/2))
        rows = math.ceil(max_elem/columns)

    ax = []
    for i in range(max_elem-1):
        state_transp = states_np[i].transpose(1, 2, 0)
        img = np.ndarray.astype(state_transp, np.uint8)
        ax.append(fig.add_subplot(rows, columns, i+1))

        info = f"rw: {float(rewards_np[i])}, act: {int(actions_np[i])}, mask: {int(masks_np[i])}, \n val: {float(values_np[i])} , gae: {round(float(gae_np[i]),2)} "
        # ax[-1].set_title(info)
        ax[-1].axis("off")
        #ax[-1].text(0.5, -0.1, "(a) my label", size=12, ha="center")
        ax[-1].text(15.0, 0.0, info, size=8, ha="center")
        plt.imshow(img)

    plt.show()
