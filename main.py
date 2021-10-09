import time

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ModelCatalog
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
from rich.progress import track

from NavigationEnv import get_env
from NavigationModel import NavModel
from Parameters import Params


def get_env_configs(params: Params):
    env_config = dict(
        N=params.agents,
        landmarks=params.landmarks,
        max_cycles=params.horizon,
        continuous_actions=False,
        name=params.env_name,
    )

    register_env(params.env_name, lambda config: get_env(config))

    params.configs['env_config'] = env_config

    return env_config


def get_policy_configs(params: Params):
    env = get_env(params.configs['env_config'])

    ModelCatalog.register_custom_model(params.model_name, NavModel)

    policies = {
        agnt: (None, env.observation_space, env.action_space, {}) for agnt in env.agents
    }
    policy_configs = dict(
        policies=policies, policy_mapping_fn=lambda agent_id: agent_id
    )

    params.configs['policy_configs'] = policy_configs

    return policy_configs


def get_model_configs(params: Params):
    model_configs = dict(
        custom_model=params.model_name,
        vf_share_layers=True,
    )
    ModelCatalog.register_custom_model(params.model_name, NavModel)

    params.configs['model_configs'] = model_configs

    return model_configs


def get_general_configs(params: Params):
    configs = dict(
        env=params.env_name,
        num_gpus=params.num_gpus if not params.debug else 0,
        num_workers=params.num_cpus if not params.debug else 0,
        framework=params.framework,
    )

    return configs


def tune_train(params: Params, configs):
    analysis = ray.tune.run(
        "PPO",
        name=f"{params.env_name}_test",
        metric="episode_reward_mean",
        config=configs,
    )
    print(analysis)


def visual_train(params: Params, config):
    PPOagent = PPOTrainer(env=params.env_name, config=config)
    # PPOagent.restore(checkpoint_path)

    env = get_env(config['env_config'])

    # building dicts
    observations = {ag: 0 for ag in env.agents}
    actions = {ag: 0 for ag in env.agents}
    rewards = {ag: 0 for ag in env.agents}
    dones = {ag: 0 for ag in env.agents}
    dones["__all__"] = False
    infos = {ag: {} for ag in env.agents}

    for ep in range(params.episodes):
        obs = env.reset()
        observations.update(obs)

        env.render()
        for _ in track(range(params.horizon), description=f"Episode {ep}..."):

            if dones['__all__']:
                break

            for agent in env.agents:
                action, _, _ = PPOagent.get_policy(agent).compute_single_action(observations[agent])
                actions[agent] = action
                obs, rew, done, info = env.step(actions)

                observations.update(obs)
                rewards.update(rew)
                dones.update(done)
                infos.update(info)

                env.render()

                time.sleep(0.1)


if __name__ == "__main__":
    params = Params()
    ray.init(local_mode=params.debug)

    # Get configs
    env_config = get_env_configs(params)
    policy_configs = get_policy_configs(params)
    model_configs = get_model_configs(params)
    configs = get_general_configs(params)

    # set specific configs in general dic
    configs["model"] = model_configs
    configs["env_config"] = env_config
    configs["multiagent"] = policy_configs

    ## TRAIING

    visual_train(params, configs)
    #tune_train(params, configs)
