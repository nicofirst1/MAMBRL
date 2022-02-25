from typing import Dict

import torch
from torch import optim, nn
from torch.nn.utils import clip_grad_norm_

from src.agent.RolloutStorage import RolloutStorage
from src.common import Params
from src.model import NextFramePredictor


def preprocess_state(state, input_noise):
    state = state.float()
    noise_prob = torch.tensor([[input_noise, 1 - input_noise]])
    noise_prob = torch.softmax(torch.log(noise_prob), dim=-1)
    noise_mask = torch.multinomial(noise_prob, state.numel(), replacement=True).view(state.shape)
    noise_mask = noise_mask.to(state)
    state = state * noise_mask + torch.median(state) * (1 - noise_mask)
    return state


class EnvModelAgent:
    """
    Container for an env model with its own optimizer
    it has both the standard trainig loop with loss function and an unsupervised rollout step
    """

    def __init__(self, agent_id: str, model: NextFramePredictor, config: Params):
        """
        @param agent_id: the agent id associated with the model
        @param model: type of model to use
        @param config:
        """
        self.agent_id = agent_id
        self.config = config
        self.device = config.device
        self.env_model = model(config)
        self.to(self.device)

        self.optimizer = optim.RMSprop(
            self.env_model.parameters(),
            lr=config.lr, eps=config.eps, alpha=config.alpha, )

    def train(self):
        self.env_model.train()

    def eval(self):
        self.env_model.eval()

    def to(self, device):
        self.env_model.to(device)
        self.device = device
        return self

    def train_step(self, rollout: RolloutStorage, frames, indices, epsilon) -> Dict:
        """
        Perform a single train step. Where N rollout steps are performed in a superviseed fashion
        @param rollout:

        @return: dictionary of metrics
        """
        action_shape = self.config.num_actions
        rollout_len = self.config.rollout_len
        c, h, w = self.config.frame_shape

        if self.config.stack_internal_states:
            self.env_model.init_internal_states(self.config.batch_size)

        n_losses = 5 if self.config.use_stochastic_model else 4
        losses = torch.empty((rollout_len, n_losses))

        actual_states = []
        predicted_frames = []

        ############################
        #   Rollout steps
        ############################
        for j in range(rollout_len):
            actions = torch.zeros((self.config.batch_size, action_shape)).to(self.config.device)
            rewards = torch.zeros((self.config.batch_size,)).to(self.config.device)
            new_states = torch.zeros((self.config.batch_size, *self.config.frame_shape)).to(self.config.device)
            values = torch.zeros((self.config.batch_size,)).to(self.config.device)

            # fill with supervised input from rollout
            for k in range(self.config.batch_size):
                actions[k] = rollout.actions[indices[k] + j]
                rewards[k] = rollout.rewards[indices[k] + j]
                new_states[k] = rollout.next_state[indices[k] + j]
                values[k] = rollout.value_preds[indices[k] + j]

            new_states_input = new_states.float()

            # predict future state
            frames_pred, reward_pred, values_pred = self.env_model(
                frames, actions, new_states_input, epsilon
            )

            # randomly use predicted or actual frame based on epsilon
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

            ######################################
            #   Logging Metrics
            ######################################
            losses[j] = torch.tensor(tab)

            losses = torch.mean(losses, dim=0)
            actual_states.append(new_states[0].detach().cpu())
            predicted_frames.append(torch.argmax(
                frames_pred[0], dim=0).detach().cpu())
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

            return metrics

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
