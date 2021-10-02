import os
import time

from NavigationEnv import get_env, raw_env

import ray
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env

#par_env = get_env()

episodes = 30
MAX_RESETS = 2

class CustomModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model = FullyConnectedNetwork(obs_space, action_space, num_outputs, model_config, name)

    def forward(self, input_dict, state, seq_lens):
        return self.model.forward(input_dict, state, seq_lens)

    def value_function(self):
        return self.model.value_function()

if __name__ == "__main__":
    ray.init(local_mode=False)
    ModelCatalog.register_custom_model("collab_nav_model", CustomModel)

    #register_env("collab_nav", get_env())

    config = {
        "env": raw_env,
        "env_config": {
            "name": "collab_nav"
        },
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "model": {
            "custom_model": "collab_nav",
            "vf_share_layers": True,
        },
        "num_workers": 1,  # parallelism
        "framework": "tf",
    }

    if True:
        ppo_config = ppo.DEFAULT_CONFIG.copy()
        ppo_config.update(config)
        ppo_config["lr"] = 1e-3

        trainer = ppo.PPOTrainer(config=ppo_config, env=raw_env)

        for _ in range(episodes):
            result = trainer.train()
            print(pretty_print(result))

            if result["timesteps_total"] >= 100000 or result["episode_reward_mean"] >= 0.1:
                break
    else:
        # Automatic learning with tune to be included
        pass

# for n_resets in range(MAX_RESETS):
#     obs = par_env.reset()
#     par_env.render()

#     done = {agent: False for agent in par_env.agents}
#     live_agents = par_env.agents[:]
#     has_finished = set()
#     for i in range(episodes):
#         actions = {
#             agent: space.sample()
#             for agent, space in par_env.action_spaces.items()
#             if agent in done and not done[agent]
#         }
#         obs, rew, done, info = par_env.step(actions)
#         par_env.render()
#         time.sleep(0.1)

#         for agent, d in done.items():
#             if d:
#                 live_agents.remove(agent)
#         has_finished |= {agent for agent, d in done.items() if d}
#         if not par_env.agents:
#             assert has_finished == set(
#                 par_env.possible_agents
#             ), "not all agents finished, some were skipped over"
#             break
