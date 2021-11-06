import time

import ray
from env.NavEnv import get_env
from ray.rllib.agents.ppo import PPOTrainer, ppo
from ray.tune import register_env
from ray.tune.logger import pretty_print
from rich.progress import track

from src.utils import Params
from src.utils.utils import trial_name_creator


def tune_train(params: Params, configs, callbacks):
    analysis = ray.tune.run(
        ppo.PPOTrainer,
        local_dir=params.LOG_DIR,
        name=f"{params.env_name}_test",
        metric="episode_reward_mean",
        config=configs,
        trial_name_creator=trial_name_creator,
        checkpoint_freq=params.checkpoint_freq,
        keep_checkpoints_num=params.max_checkpoint_keep,
        resume=params.resume_training,
        callbacks=callbacks,
    )

    checkpoints = analysis.get_trial_checkpoints_paths(trial=analysis.get_best_trial('episode_reward_mean'),
                                                       metric='episode_reward_mean')
    # retriev the checkpoint path; we only have a single checkpoint, so take the first one
    checkpoint_path = checkpoints[0][0]
    return checkpoint_path, analysis


def load(self, path):
    """
    Load a trained RLlib agent from the specified path. Call this before testing a trained agent.
    :param path: Path pointing to the agent's saved checkpoint (only used for RLlib agents)
    """
    self.agent = ppo.PPOTrainer(config=self.config, env=self.env_class)
    self.agent.restore(path)


def test(self):
    """Test trained agent for a single episode. Return the episode reward"""
    # instantiate env class
    env = self.env_class(self.env_config)

    # run until episode ends
    episode_reward = 0
    done = False
    obs = env.reset()
    while not done:
        action = self.agent.compute_action(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

    return episode_reward


def visual_train(params: Params, config, render=True):
    PPO_Agent = PPOTrainer(env=params.env_name, config=config)
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

        if render: env.render()
        for _ in track(range(params.horizon), description=f"Episode {ep}..."):

            if dones['__all__']:
                break

            for agent_id in env.agents:
                action, _, _ = PPO_Agent.get_policy(agent_id).compute_single_action(observations[agent_id])
                actions[agent_id] = action
                obs, rew, done, info = env.step(actions)

                observations.update(obs)
                rewards.update(rew)
                dones.update(done)
                infos.update(info)

                if render:
                    env.render()

                    time.sleep(0.1)


def cnn_train(params: Params, config):
    PPO_Agent = PPOTrainer(env=params.env_name, config=config)
    # agent.restore(checkpoint_path)

    env = get_env(config['env_config'])

    # building dicts
    actions = {ag: 0 for ag in env.agents}
    rewards = {ag: 0 for ag in env.agents}
    dones = {ag: 0 for ag in env.agents}
    dones["__all__"] = False
    infos = {ag: {} for ag in env.agents}

    for ep in range(params.episodes):
        # reset envs
        _ = env.reset()
        # call render and set image as observation value
        img = env.render(mode="rgb_array")

        for _ in track(range(params.horizon), description=f"Episode {ep}..."):

            if dones['__all__']:
                break

            for agent_id in env.agents:
                action, _, _ = PPO_Agent.get_policy(agent_id).compute_single_action(img)
                actions[agent_id] = action

                _, rew, done, info = env.step(actions)
                img = env.render(mode="rgb_array")

                rewards.update(rew)
                dones.update(done)
                infos.update(info)


def agent_train(params: Params, config):
    register_env(params.env_name,
                 lambda config: get_env(config))

    agent = PPOTrainer(env=params.env_name, config=config)
    # agent.restore(checkpoint_path)

    ep = 0
    for ep in track(range(params.episodes), description=f"Episode {ep}..."):
        result = agent.train()
        pretty_print(result)
