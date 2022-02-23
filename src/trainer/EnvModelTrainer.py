import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_
from tqdm import trange

from src.agent.RolloutStorage import RolloutStorage
from src.common import Params, mas_dict2tensor
from src.common.utils import one_hot_encode
from src.env.EnvWrapper import get_env_wrapper
from src.model import NextFramePredictor
from src.trainer.BaseTrainer import BaseTrainer
from src.trainer.Policies import OptimalAction


def preprocess_state(state, input_noise):
    state = state.float() / 255
    noise_prob = torch.tensor([[input_noise, 1 - input_noise]])
    noise_prob = torch.softmax(torch.log(noise_prob), dim=-1)
    noise_mask = torch.multinomial(noise_prob, state.numel(), replacement=True).view(state.shape)
    noise_mask = noise_mask.to(state)
    state = state * noise_mask + torch.median(state) * (1 - noise_mask)
    return state


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


class EnvModelContainer:
    """
    Container for an env model with its own optimizer
    it has both the standard trainig loop with loss function and an unsupervised rollout step
    """

    def __init__(self, agent_id: str, model: NextFramePredictor, config: Params, logging_fn=None):
        """
        @param agent_id: the agent id associated with the model
        @param model: type of model to use
        @param config:
        @param logging_fn: the "on_batch_end" function of the envLogger, used to log batches
        """
        self.agent_id = agent_id
        self.config = config
        self.device = config.device
        self.env_model = model(config)
        self.to(self.device)
        self.logging_fn = logging_fn

        self.optimizer = optim.RMSprop(
            self.env_model.parameters(),
            lr=config.lr, eps=config.eps, alpha=config.alpha, )

    def to(self, device):
        self.env_model.to(device)
        self.device = device
        return self

    def train(self, rollout: RolloutStorage):
        """
        Training loop with supervised rollout steps
        @param rollout:
        @return:
        """
        c, h, w = self.config.frame_shape
        rollout_len = self.config.rollout_len
        states, actions, rewards, new_states, _, values = rollout.get(0)

        action_shape = self.config.num_actions
        new_state_shape = new_states.shape

        ## fixme: nel rollout non può mai verificarsi questa situazione
        if values is None:
            raise BufferError(
                "Can't train the world model, the buffer does not contain one full episode."
            )

        assert states.dtype == torch.uint8
        assert actions.dtype == torch.int64
        assert rewards.dtype == torch.float32
        assert new_states.dtype == torch.uint8
        assert values.dtype == torch.float32

        self.env_model.train()

        iterator = trange(
            0, self.config.env_model_steps, rollout_len, desc="Training world model", unit_scale=rollout_len
        )
        for i in iterator:
            decay_steps = self.config.scheduled_sampling_decay_steps
            inv_base = torch.exp(torch.log(torch.tensor(0.01)) / (decay_steps // 4))
            epsilon = inv_base ** max(decay_steps // 4 - i, 0)
            progress = min(i / decay_steps, 1)
            progress = progress * (1 - 0.01) + 0.01
            epsilon *= progress
            epsilon = 1 - epsilon

            indices = get_indices(rollout, self.config.batch_size, rollout_len)
            frames = torch.zeros((self.config.batch_size, c * self.config.num_frames, h, w))
            frames = frames.to(self.config.device)

            for j in range(self.config.batch_size):
                frames[j] = rollout.states[indices[j]].clone()
            frames = preprocess_state(frames, self.config.input_noise)

            n_losses = 5 if self.config.use_stochastic_model else 4
            losses = torch.empty((rollout_len, n_losses))

            if self.config.stack_internal_states:
                self.env_model.init_internal_states(self.config.batch_size)

            actual_states = []
            predicted_frames = []
            for j in range(rollout_len):
                actions = torch.zeros((self.config.batch_size, action_shape)).to(self.config.device)
                rewards = torch.zeros((self.config.batch_size,)).to(self.config.device)
                new_states = torch.zeros((self.config.batch_size, *new_state_shape)).to(self.config.device)
                values = torch.zeros((self.config.batch_size,)).to(self.config.device)

                for k in range(self.config.batch_size):
                    actions[k] = rollout.actions[indices[k] + j]
                    rewards[k] = rollout.rewards[indices[k] + j]
                    new_states[k] = rollout.next_state[indices[k] + j]
                    values[k] = rollout.value_preds[indices[k] + j]

                new_states_input = new_states.float() / 255
                frames_pred, reward_pred, values_pred = self.env_model(
                    frames, actions, new_states_input, epsilon
                )

                actual_states.append(new_states[0].detach().cpu())
                predicted_frames.append(torch.argmax(
                    frames_pred[0], dim=0).detach().cpu())

                if j < rollout_len - 1:
                    for k in range(self.config.batch_size):
                        if float(torch.rand((1,))) < epsilon:
                            frame = new_states[k]
                        else:
                            frame = torch.argmax(frames_pred[k], dim=0)

                        frame = preprocess_state(frame, self.config.input_noise)
                        frames[k] = torch.cat((frames[k, c:], frame), dim=0)

                # given prediction and values performa a backward pass on loss
                tab = self.loss_step(frames_pred, reward_pred, values_pred, new_states, rewards, values)
                losses[j] = torch.tensor(tab)

            losses = torch.mean(losses, dim=0)
            metrics = {
                "loss": float(losses[0]),
                "loss_reconstruct": float(losses[1]),
                "loss_value": float(losses[2]),
                "loss_reward": float(losses[3]),
                "imagined_state": predicted_frames,
                "actual_state": actual_states,
                "epsilon": epsilon
            }

            if self.config.use_stochastic_model:
                metrics.update({"loss_lstm": float(losses[4])})

            if self.logging_fn is not None:
                metrics = {self.agent_id: metrics}
                self.logging_fn(metrics, batch_id=i, is_training=True)

    def loss_step(self, frames_pred, reward_pred, values_pred, frames, rewards, values):
        """
        Perform a single loss backward step given the predicted and real values
        @param frames_pred:
        @param reward_pred:
        @param values_pred:
        @param frames:
        @param rewards:
        @param values:
        @return: list of loss values used for logs
        """
        loss_reconstruct = nn.CrossEntropyLoss(reduction="none")(
            frames_pred, frames.long()
        )

        clip = torch.tensor(self.config.target_loss_clipping).to(
            self.config.device
        )

        loss_reconstruct = torch.max(loss_reconstruct, clip)
        loss_reconstruct = (
                loss_reconstruct.mean() - self.config.target_loss_clipping
        )

        reward_pred = reward_pred.squeeze()
        loss_value = nn.MSELoss()(values_pred, values)
        loss_reward = nn.MSELoss()(reward_pred, rewards)
        loss = loss_reconstruct + loss_value + loss_reward

        # save for logs
        tab = [
            float(loss),
            float(loss_reconstruct),
            float(loss_value),
            float(loss_reward),
        ]

        if self.config.use_stochastic_model:
            loss_lstm = self.env_model.stochastic_model.get_lstm_loss()
            loss = loss + loss_lstm
            tab.append(float(loss_lstm))

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.env_model.parameters(), self.config.clip_grad_norm)
        self.optimizer.step()

        return tab

    def rollout_steps(self, frames: torch.Tensor, actions: torch.Tensor, act_fn):
        """
        Perform N rollout steps in an unsupervised fashion (no rolloutStorage)
        Bs=batch_size
        @param frames: [Bs, C, W, H] tensor of frames where the last one is the new observation
        @param actions: [Bs, num actions] tensor with a one hot encoding of the chosen action
        @param act_fn: function taking as input the observation and returning an action
        @return:
            predicted observation : [N, Bs, C, W, H]: tensor for predicted observations
            predicted rewards : [N, Bs, 1]
        """

        batch_size = frames.shape[0]
        self.env_model.eval()
        self.env_model.init_internal_states(batch_size)

        new_obs = frames
        pred_obs = []
        pred_rews = []

        for j in range(self.config.rollout_len):
            # update frame and actions
            frames = torch.concat([frames, new_obs], dim=0)
            frames = frames[1:]

            # get new action given pred frame with policy
            new_action, _, _ = act_fn(observation=new_obs)
            new_action = one_hot_encode(new_action, self.config.num_actions)
            new_action = new_action.to(self.config.device).unsqueeze(dim=0)

            actions = torch.concat([actions, new_action], dim=0).float()
            actions = actions[1:]

            with torch.no_grad():
                new_obs, pred_rew, pred_values = self.env_model(frames, actions)

            # remove feature dimension
            new_obs = torch.argmax(new_obs, dim=1)

            # append to pred list
            pred_obs.append(new_obs)
            pred_rews.append(pred_rew)

            # get last, normalize and add fake batch dimension for stack
            new_obs = new_obs[-1] / 255
            new_obs = new_obs.unsqueeze(dim=0)

        # todo: should pred_obs be normalized?
        pred_obs = torch.stack(pred_obs) / 255
        pred_rews = torch.stack(pred_rews)

        pred_obs = pred_obs.to(self.device)
        pred_rews = pred_rews.to(self.device)

        return pred_obs, pred_rews


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

        logging_fn = None
        if self.config.use_wandb and False:
            from logging_callbacks import EnvModelWandb
            self.logger = EnvModelWandb(
                train_log_step=20,
                val_log_step=50,
                project="env_model",
            )
            logging_fn = self.logger.on_batch_end

        self.env_model = {k: EnvModelContainer(k, model, config, logging_fn) for k in self.cur_env.agents}

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

        metrics = {}
        for ag, model in self.env_model.items():
            metrics[ag] = model.train(rollout)


if __name__ == "__main__":
    params = Params()
    env = get_env_wrapper(params)

    trainer = EnvModelTrainer(NextFramePredictor, env, params)
    for step in trange(params.env_model_epochs, desc="Training env model"):
        rollout = trainer.collect_trajectories()
        trainer.train(rollout)
