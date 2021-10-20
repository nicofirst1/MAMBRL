import time

import ray
from ray.rllib.agents.ppo import PPOTrainer
from ray.tune import register_env
from ray.tune.logger import pretty_print
from rich.progress import track

from env.NavEnv import get_env
from src.utils import Params
from src.utils.utils import trial_name_creator


def tune_train(params: Params, configs):
    analysis = ray.tune.run(
        "PPO",
        local_dir=params.LOG_DIR,
        name=f"{params.env_name}_test",
        metric="episode_reward_mean",
        config=configs,
        trial_name_creator=trial_name_creator,
        checkpoint_freq=params.checkpoint_freq,
        keep_checkpoints_num=params.max_checkpoint_keep,
        resume=params.resume_training
    )
    print(analysis)


def visual_train(params: Params, config):
    agent = PPOTrainer(env=params.env_name, config=config)
    # agent.restore(checkpoint_path)

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
                action, _, _ = agent.get_policy(agent).compute_single_action(observations[agent])
                actions[agent] = action
                obs, rew, done, info = env.step(actions)

                observations.update(obs)
                rewards.update(rew)
                dones.update(done)
                infos.update(info)

                env.render()

                time.sleep(0.1)


def agent_train(params: Params, config):
    register_env(params.env_name,
                 lambda config: get_env(config))

    agent = PPOTrainer(env=params.env_name, config=config)
    # agent.restore(checkpoint_path)

    ep=0
    for ep in track(range(params.episodes), description=f"Episode {ep}..."):
        result = agent.train()
        pretty_print(result)
