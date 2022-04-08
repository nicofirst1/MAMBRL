import os.path

import torch
from tqdm import trange

from src.agent.EnvModelAgent import EnvModelAgent, preprocess_state
from src.agent.RolloutStorage import RolloutStorage
from src.common import Params, mas_dict2tensor
from src.env.EnvWrapper import get_env_wrapper
from src.model import NextFramePredictor
from src.trainer.BaseTrainer import BaseTrainer
from src.trainer.Policies import OptimalAction


def get_indices(rollout, batch_size, rollout_len):
    def get_index():
        index = -1
        while index == -1:
            index = int(torch.randint(rollout.rewards.size(0) - rollout_len, size=(1,)))
            for i in range(rollout_len):
                ## fixme: controllo sul value che va rivisto sempre perché non può essere None
                # if not rollout.masks[index + 1] or rollout.value_preds[index + i] is None:
                if not rollout.masks[index + 1]:
                    index = -1
                    break
        return index

    return [get_index() for _ in range(batch_size)]


class EnvModelTrainer(BaseTrainer):
    def __init__(self, model: NextFramePredictor, env, config: Params):
        """__init__ module.

        Parameters
        ----------
        model : NextFramePredictor
            model in src.model.EnvModel
        env : env class
            one of the env classes defined in the src.env directory
        config : Params
            instance of the class Params defined in src.common.Params

        Returns
        -------
        None.

        """
        super(EnvModelTrainer, self).__init__(env, config)

        self.policy = OptimalAction(self.cur_env, config.num_actions, config.device)

        if self.config.use_wandb:
            from logging_callbacks import EnvModelWandb
            self.logger = EnvModelWandb(
                train_log_step=20,
                val_log_step=50,
                project="env_model",
            )

        self.env_model = {k: EnvModelAgent(k, model, config) for k in self.cur_env.agents}

        self.config = config

    def collect_trajectories(self) -> RolloutStorage:
        rollout = RolloutStorage(
            num_steps=self.config.horizon * self.config.episodes,
            frame_shape=self.config.frame_shape,
            obs_shape=self.config.obs_shape,
            num_actions=self.config.num_actions,
            num_agents=1,
        )
        rollout.to(self.config.device)

        if self.logger is not None:
            self.logger.epoch += 1

        action_dict = {agent_id: None for agent_id in self.cur_env.agents}
        done = {agent_id: None for agent_id in self.cur_env.agents}

        for episode in trange(self.config.episodes, desc="Collecting trajectories.."):
            done["__all__"] = False
            observation = self.cur_env.reset()
            rollout.states[episode * self.config.horizon] = observation.unsqueeze(dim=0)

            for step in range(self.config.horizon):
                observation = observation.unsqueeze(dim=0).to(self.config.device)

                for agent_id in self.cur_env.agents:
                    with torch.no_grad():
                        action, _, _ = self.policy.act(agent_id, observation)
                        action_dict[agent_id] = action

                observation, rewards, done, _ = self.cur_env.step(action_dict)

                actions = mas_dict2tensor(action_dict, int)
                rewards = mas_dict2tensor(rewards, float)
                masks = (~torch.tensor(done["__all__"])).float().unsqueeze(0)

                rollout.insert(
                    state=observation,
                    next_state=observation[-3:, :, :],
                    action=actions,
                    action_log_probs=None,
                    value_preds=None,
                    reward=rewards,
                    mask=masks
                )

                if done["__all__"]:
                    rollout.compute_value_world_model(episode * self.config.horizon + step, self.config.gamma)
                    observation = self.cur_env.reset()
        return rollout

    def train(self, rollout: RolloutStorage):

        """
            Training loop with supervised rollout steps
            @param rollout:
            @return:
        """
        c, h, w = self.config.frame_shape
        rollout_len = self.config.rollout_len
        states, actions, rewards, new_states, _, values = rollout.get(0)

        assert states.dtype == torch.float32
        assert actions.dtype == torch.int64
        assert rewards.dtype == torch.float32
        assert new_states.dtype == torch.float32
        assert values.dtype == torch.float32

        iterator = trange(
            0, self.config.env_model_steps, rollout_len, desc="Training world model", unit_scale=rollout_len
        )

        metrics = {}

        for i in iterator:
            # Define Epsilon
            decay_steps = self.config.scheduled_sampling_decay_steps
            inv_base = torch.exp(torch.log(torch.tensor(0.01)) / (decay_steps // 4))
            epsilon = inv_base ** max(decay_steps // 4 - i, 0)
            progress = min(i / decay_steps, 1)
            progress = progress * (1 - 0.01) + 0.01
            epsilon *= progress
            epsilon = 1 - epsilon

            # init indices and frames
            indices = get_indices(rollout, self.config.batch_size, rollout_len)
            frames = torch.zeros((self.config.batch_size, c * self.config.num_frames, h, w))
            frames = frames.to(self.config.device)

            # populate frames with rollout states
            for j in range(self.config.batch_size):
                frames[j] = rollout.states[indices[j]].clone()
            frames = preprocess_state(frames, self.config.input_noise)

            # train and log
            for ag, model in self.env_model.items():
                model.train()
                metrics[ag] = model.train_step(rollout, frames.clone(), indices, epsilon)

        return metrics

    def checkpoint(self, path):
        self.env_model["agent_0"].env_model.save_model(os.path.join(path, "env_model.pt"))

    def restore_training(self):
        pass


if __name__ == "__main__":
    params = Params()
    env = get_env_wrapper(params)

    trainer = EnvModelTrainer(NextFramePredictor, env, params)
    for epoch in trange(params.env_model_epochs, desc="Training env model"):
        rollout = trainer.collect_trajectories()
        logs = trainer.train(rollout)

        if params.use_wandb and epoch % params.log_step == 0:
            # fixme: HARDCODED agent_0
            trainer.logger.on_batch_end(logs["agent_0"], batch_id=epoch, is_training=True)

        if epoch % 400 == 0:
            trainer.checkpoint(params.WEIGHT_DIR)
