"""rollout_test file."""
import random
import torch
from src.env.NavEnv import get_env
from src.common.utils import get_env_configs, mas_dict2tensor, order_state
from src.common.Params import Params
from src.model.RolloutStorage import RolloutStorage

params = Params()
params.resize = False
params.gray_scale = False
device = params.device
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

current_state = env.reset()
current_state = order_state(current_state)


size_minibatch = params.minibatch  # 2
epochs = params.epochs  # 1
steps_per_episode = params.horizon  # params.horizon
number_of_episodes = 10  # int(10e5)

dones = {agent_id: False for agent_id in env.agents}
action_dict = {agent_id: False for agent_id in env.agents}
values_dict = {agent_id: False for agent_id in env.agents}

counter = 0
for step in range(steps_per_episode):
    action_log_probs_list = []
    current_state = current_state.to(params.device).unsqueeze(dim=0)

    # let every agent act
    for agent_id in env.agents:

        # skip action for done agents
        if dones[agent_id]:
            action_dict[agent_id] = None
            continue
        action_dict[agent_id] = random.randint(0, num_actions-1)

    next_state, rewards, dones, _ = env.step(action_dict)

    if dones.pop("__all__"):
        break

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
