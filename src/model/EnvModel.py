import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import truncnorm

from src.common.utils import one_hot_encode
from .utils import (
    ActionInjector,
    Container,
    MeanAttention,
    bit_to_int,
    get_timing_signal_nd,
    int_to_bit,
    mix,
    sample_with_temperature,
    standardize_frame,
)


class RewardEstimator(nn.Module):
    def __init__(self, config, input_size):
        super().__init__()
        self.config = config

        self.dense1 = nn.Linear(input_size, 128)
        self.dense2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.dense1(x)
        x = F.relu(x)
        x = self.dense2(x)
        return x


class ValueEstimator(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.dense = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.dense(x)


class MiddleNetwork(nn.Module):
    def __init__(self, config, filters):
        super().__init__()
        self.config = config

        self.middle_network = []
        for i in range(self.config.hidden_layers):
            self.middle_network.append(
                nn.Conv2d(filters, filters, 3, padding=1))
            if i == 0:
                self.middle_network.append(None)
            else:
                self.middle_network.append(
                    nn.InstanceNorm2d(filters, affine=True, eps=1e-6)
                )
        self.middle_network = nn.ModuleList(self.middle_network)

    def forward(self, x):
        for i in range(self.config.hidden_layers):
            y = F.dropout(x, self.config.residual_dropout)
            y = self.middle_network[2 * i](y)  # Conv
            y = F.relu(y)

            if i == 0:
                x = y
            else:
                x = self.middle_network[2 * i + 1](x + y)  # LayerNorm

        return x


class BitsPredictor(nn.Module):
    def __init__(
            self, config, input_size, state_size, total_number_bits, bits_at_once=8
    ):
        super().__init__()
        self.config = config
        self.total_number_bits = total_number_bits
        self.bits_at_once = bits_at_once
        self.dense1 = nn.Linear(input_size, state_size)
        self.dense2 = nn.Linear(input_size, state_size)
        self.dense3 = nn.Linear(input_size, state_size)
        self.dense4 = nn.Linear(2 ** bits_at_once, state_size)
        self.dense5 = nn.Linear(state_size, 2 ** bits_at_once)
        self.lstm = nn.LSTMCell(state_size, state_size)

    def forward(self, x, temperature, target_bits=None):
        x = torch.flatten(x, start_dim=1)

        first_lstm_input = self.dense1(x)
        h_state = self.dense2(x)
        c_state = self.dense3(x)

        if target_bits is not None:
            target_bits = target_bits.view(
                (-1, self.total_number_bits // self.bits_at_once, self.bits_at_once)
            )
            target_bits = torch.max(
                target_bits, torch.tensor(0.0).to(self.config.device)
            )
            target_ints = bit_to_int(target_bits, self.bits_at_once).long()
            target_hot = one_hot_encode(
                target_ints, 2 ** self.bits_at_once, dtype=torch.float32
            )
            target_embedded = self.dense4(target_hot)
            target_embedded = F.dropout(target_embedded, 0.1)
            teacher_input = torch.cat(
                (first_lstm_input.unsqueeze(1), target_embedded), dim=1
            )

            outputs = []
            for i in range(self.total_number_bits // self.bits_at_once):
                lstm_input = teacher_input[:, i, :]
                h_state, c_state = self.lstm(lstm_input, (h_state, c_state))
                outputs.append(h_state)
            outputs = torch.stack(outputs, dim=1)
            outputs = F.dropout(outputs, 0.1)
            pred = self.dense5(outputs)

            loss = nn.CrossEntropyLoss()(pred.permute((0, 2, 1)), target_ints)

            return pred, loss / self.config.horizon

        outputs = []
        lstm_input = first_lstm_input
        for i in range(self.total_number_bits // self.bits_at_once):
            h_state, c_state = self.lstm(lstm_input, (h_state, c_state))
            discrete_logits = self.dense5(h_state)
            discrete_samples = sample_with_temperature(
                discrete_logits, temperature)
            outputs.append(discrete_samples)
            lstm_input = self.dense4(
                one_hot_encode(discrete_samples, 256, dtype=torch.float32)
            )
        outputs = torch.stack(outputs, dim=1)
        outputs = int_to_bit(outputs, self.bits_at_once)
        outputs = outputs.view((-1, self.total_number_bits))

        return 2 * outputs - 1, 0.0


class StochasticModel(nn.Module):
    def __init__(self, config, layer_shape, n_action):
        super().__init__()
        self.config = config
        channels = self.config.frame_shape[0] * (self.config.num_frames + 1)
        filters = [128, 512]
        self.lstm_loss = None
        self.get_lstm_loss()

        self.timing_signal = get_timing_signal_nd(
            (self.config.hidden_size, *self.config.obs_shape[1:])
        ).to(self.config.device)

        self.input_embedding = nn.Conv2d(channels, self.config.hidden_size, 1)
        self.conv1 = nn.Conv2d(self.config.hidden_size,
                               filters[0], 8, 4, padding=2)
        self.conv2 = nn.Conv2d(filters[0], filters[1], 8, 4, padding=2)
        self.dense1 = nn.Linear(2 * 2 * channels, self.config.bottleneck_bits)
        self.dense2 = nn.Linear(self.config.bottleneck_bits, layer_shape[0])
        self.dense3 = nn.Linear(self.config.bottleneck_bits, layer_shape[0])

        self.action_injector = ActionInjector(
            n_action, self.config.hidden_size)
        self.mean_attentions = nn.ModuleList(
            [MeanAttention(n_filter, 2 * channels) for n_filter in filters]
        )
        self.bits_predictor = BitsPredictor(
            config,
            layer_shape[0] * layer_shape[1] * layer_shape[2],
            self.config.latent_state_size,
            self.config.bottleneck_bits,
        )

    def add_bits(self, layer, bits):
        z_mul = self.dense2(bits)
        z_mul = torch.sigmoid(z_mul)
        z_add = self.dense3(bits)
        z_mul = z_mul.unsqueeze(-1).unsqueeze(-1)
        z_add = z_add.unsqueeze(-1).unsqueeze(-1)
        return layer * z_mul + z_add

    def forward(self, layer, inputs, action, target, epsilon):
        if self.training and target is not None:
            x = torch.cat((inputs, target), dim=1)
            x = self.input_embedding(x)
            x = x + self.timing_signal

            x = self.action_injector(x, action)

            x = self.conv1(x)
            x1 = self.mean_attentions[0](x)
            x = F.relu(x)
            x = self.conv2(x)
            x2 = self.mean_attentions[1](x)

            x = torch.cat((x1, x2), dim=-1)
            x = self.dense1(x)

            bits_clean = (2 * (0 < x).float()).detach() - 1
            truncated_normal = truncnorm.rvs(-2, 2, size=x.shape, scale=0.2)
            truncated_normal = torch.tensor(truncated_normal, dtype=torch.float32).to(
                self.config.device
            )
            x = x + truncated_normal
            x = torch.tanh(x)
            bits = x + (2 * (0 < x).float() - 1 - x).detach()
            noise = torch.rand_like(x)
            noise = 2 * (self.config.bottleneck_noise < noise).float() - 1
            bits = bits * noise
            bits = mix(bits, x, 1 - epsilon)

            _, lstm_loss = self.bits_predictor(layer, 1.0, bits_clean)
            self.lstm_loss = self.lstm_loss + lstm_loss

            bits_pred, _ = self.bits_predictor(layer, 1.0)
            bits_pred = bits_clean + (bits_pred - bits_clean).detach()
            bits = mix(
                bits_pred, bits, 1 - (1 - epsilon) *
                self.config.latent_rnn_max_sampling
            )

            res = self.add_bits(layer, bits)
            return mix(
                res, layer, 1 - (1 - epsilon) *
                self.config.latent_use_max_probability
            )

        bits, _ = self.bits_predictor(layer, 1.0)
        return self.add_bits(layer, bits)

    def get_lstm_loss(self, reset=True):
        res = self.lstm_loss
        if reset:
            self.lstm_loss = torch.tensor(0.0).to(self.config.device)
        return res


class NextFramePredictor(Container):
    def __init__(self, config):
        # fixme: remove config and use custom dict
        super().__init__()
        self.config = config
        filters = self.config.hidden_size

        # Internal states

        channels = self.config.obs_shape[0]
        if self.config.stack_internal_states:
            channels += self.config.recurrent_state_size

        self.internal_states = None
        self.last_x_start = None
        if self.config.stack_internal_states:
            self.gate = nn.Conv2d(
                channels, 2 * self.config.recurrent_state_size, 3, padding=1
            )

        # Model
        self.timing_signals = []
        self.input_embedding = nn.Conv2d(channels, self.config.hidden_size, 1)
        nn.init.normal_(self.input_embedding.bias, std=0.01)

        self.downscale_layers = []
        shape = [self.config.hidden_size, *self.config.obs_shape[1:]]
        self.timing_signals.append(
            get_timing_signal_nd(shape).to(self.config.device))
        shapes = [shape]
        for i in range(self.config.compress_steps):
            in_filters = filters
            if i < self.config.filter_double_steps:
                filters *= 2

            self.timing_signals.append(
                get_timing_signal_nd(shape).to(self.config.device)
            )
            shape = [filters, shape[1] // 2, shape[2] // 2]
            shapes.append(shape)

            self.downscale_layers.append(
                nn.Conv2d(in_filters, filters, 4, stride=2, padding=1)
            )
            self.downscale_layers.append(
                nn.InstanceNorm2d(filters, affine=True, eps=1e-6)
            )

        self.downscale_layers = nn.ModuleList(self.downscale_layers)

        middle_shape = shape

        self.upscale_layers = []
        self.action_injectors = [ActionInjector(
            self.config.num_actions, filters)]
        for i in range(self.config.compress_steps):
            self.action_injectors.append(
                ActionInjector(self.config.num_actions, filters)
            )

            in_filters = filters
            if i >= self.config.compress_steps - self.config.filter_double_steps:
                filters //= 2

            shape = [filters, shape[1] * 2, shape[2] * 2]
            output_padding = (
                0 if shape[1] == shapes[-i - 2][1] else 1,
                0 if shape[2] == shapes[-i - 2][2] else 1,
            )
            shape = [
                filters,
                shape[1] + output_padding[0],
                shape[2] + output_padding[1],
            ]

            self.upscale_layers.append(
                nn.ConvTranspose2d(
                    in_filters,
                    filters,
                    4,
                    stride=2,
                    padding=1,
                    output_padding=output_padding,
                )
            )
            self.upscale_layers.append(
                nn.InstanceNorm2d(filters, affine=True, eps=1e-6)
            )
            self.timing_signals.append(
                get_timing_signal_nd(shape).to(self.config.device)
            )

        self.upscale_layers = nn.ModuleList(self.upscale_layers)
        self.action_injectors = nn.ModuleList(self.action_injectors)

        self.logits = nn.Conv2d(
            self.config.hidden_size, 256 * self.config.frame_shape[0], 1
        )

        # Sub-models
        self.middle_network = MiddleNetwork(self.config, middle_shape[0])
        self.reward_estimator = RewardEstimator(
            self.config, middle_shape[0] + filters)
        self.value_estimator = ValueEstimator(
            middle_shape[0] * middle_shape[1] * middle_shape[2]
        )
        self.stochastic_model = StochasticModel(
            self.config, middle_shape, self.config.num_actions
        )

        if self.config.stack_internal_states:
            self.init_internal_states(self.config.batch_size)

    def to(self, device):
        self.action_injectors = self.action_injectors.to(device)
        self.downscale_layers = self.downscale_layers.to(device)
        self.gate = self.gate.to(device)
        self.input_embedding = self.input_embedding.to(device)
        self.logits = self.logits.to(device)
        self.middle_network = self.middle_network.to(device)
        self.reward_estimator = self.reward_estimator.to(device)
        self.stochastic_model = self.stochastic_model.to(device)
        self.upscale_layers = self.upscale_layers.to(device)
        self.value_estimator = self.value_estimator.to(device)

        return self

    def save_model(self, path: str):
        torch.save({
            "action_injectors": self.action_injectors.state_dict(),
            "downscale_layers": self.downscale_layers.state_dict(),
            "gate": self.gate.state_dict(),
            "input_embedding": self.input_embedding.state_dict(),
            "logits": self.logits.state_dict(),
            "middle_network": self.middle_network.state_dict(),
            "reward_estimator": self.reward_estimator.state_dict(),
            "stochastic_model": self.stochastic_model.state_dict(),
            "upscale_layers": self.upscale_layers.state_dict(),
            "value_estimator": self.value_estimator.state_dict()
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path)

        self.action_injectors.load_state_dict(checkpoint["action_injectors"])
        self.downscale_layers.load_state_dict(checkpoint["downscale_layers"])
        self.gate.load_state_dict(checkpoint["gate"])
        self.input_embedding.load_state_dict(checkpoint["input_embedding"])
        self.logits.load_state_dict(checkpoint["logits"])
        self.middle_network.load_state_dict(checkpoint["middle_network"])
        self.reward_estimator.load_state_dict(checkpoint["reward_estimator"])
        self.stochastic_model.load_state_dict(checkpoint["stochastic_model"])
        self.upscale_layers.load_state_dict(checkpoint["upscale_layers"])
        self.value_estimator.load_state_dict(checkpoint["value_estimator"])

    def init_internal_states(self, batch_size):
        self.internal_states = torch.zeros(
            (batch_size, self.config.recurrent_state_size,
             *self.config.obs_shape[1:])
        ).to(self.config.device)
        self.last_x_start = None

    def get_internal_states(self):
        internal_states = self.internal_states
        if self.last_x_start is None:
            return internal_states
        state_activation = torch.cat(
            (internal_states, self.last_x_start), dim=1)
        state_gate_candidate = self.gate(state_activation)
        state_gate, state_candidate = torch.split(
            state_gate_candidate, self.config.recurrent_state_size, dim=1
        )
        state_gate = torch.sigmoid(state_gate)
        state_candidate = torch.tanh(state_candidate)
        internal_states = internal_states * state_gate
        internal_states = internal_states + state_candidate * (1 - state_gate)
        self.internal_states = internal_states.detach()
        return internal_states

    def forward(self, x, action, target=None, epsilon=0):
        # Normalize Input
        x = x / 255.0

        # todo: add function definition
        x_start = torch.stack([standardize_frame(frame) for frame in x])

        # fixme: qui qualcosa non quadra con le dimensioni, quindi per ora è disabilitato
        # FIXME: il problema è che internal_state dipende dalle dimensioni del batch
        if self.config.stack_internal_states:
            # internal_states = self.get_internal_states()
            # FIXME: HARDCODE, to remove
            internal_states = self.internal_states
            internal_states = internal_states[-1].unsqueeze(dim=0)
            x = torch.cat((x_start, internal_states), dim=1)
            self.last_x_start = x_start
        else:
            x = x_start

        x = self.input_embedding(x)
        x = x + self.timing_signals[0]

        inputs = []
        for i in range(self.config.compress_steps):
            inputs.append(x)
            x = F.dropout(x, self.config.dropout)
            x = x + self.timing_signals[i + 1]
            x = self.downscale_layers[2 * i](x)  # Conv
            x = F.relu(x)
            x = self.downscale_layers[2 * i + 1](x)  # LayerNorm

        value_pred = self.value_estimator(torch.flatten(x, start_dim=1)).squeeze(-1)

        x = self.action_injectors[0](x, action)

        if target is not None:
            for batch_index in range(len(target)):
                target[batch_index] = standardize_frame(target[batch_index])

        x = self.stochastic_model(x, x_start, action, target, epsilon)

        x_mid = torch.mean(x, dim=(2, 3))
        x = self.middle_network(x)

        inputs = list(reversed(inputs))
        for i in range(self.config.compress_steps):
            x = F.dropout(x, self.config.dropout)
            x = self.action_injectors[i + 1](x, action)
            x = self.upscale_layers[2 * i](x)  # ConvTranspose
            x = F.relu(x)
            x = x + inputs[i]
            x = self.upscale_layers[2 * i + 1](x)  # LayerNorm
            x = x + self.timing_signals[1 + self.config.compress_steps + i]

        x_fin = torch.mean(x, dim=(2, 3))

        reward_pred = self.reward_estimator(torch.cat((x_mid, x_fin), dim=1))

        x = self.logits(x)
        x = x.view((-1, 256, *self.config.frame_shape))
        return x, reward_pred, value_pred

    def rollout_steps(self, frames: torch.Tensor, act_fn):
        """
        Perform N rollout steps in an unsupervised fashion (no rolloutStorage)
        Bs=batch_size
        @param frames: [Bs, C, W, H] tensor of frames where the last one is the new observation
        @param act_fn: function taking as input the observation and returning an action
        @return:
            predicted observation : [N, Bs, C, W, H]: tensor for predicted observations
            predicted rewards : [N, Bs, 1]
        """

        batch_size = frames.shape[0]
        self.init_internal_states(batch_size)

        actions = torch.zeros((batch_size, self.config.num_actions))

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
                new_obs, pred_rew, pred_values = self.forward(frames, actions)

            # remove feature dimension
            new_obs = torch.argmax(new_obs, dim=1)

            # append to pred list
            pred_obs.append(new_obs)
            pred_rews.append(pred_rew)

            # get last, normalize and add fake batch dimension for stack
            new_obs = new_obs[-1] / 255
            new_obs = new_obs.unsqueeze(dim=0)

        pred_obs = torch.stack(pred_obs) / 255
        pred_rews = torch.stack(pred_rews)

        pred_obs = pred_obs.to(self.device)
        pred_rews = pred_rews.to(self.device)

        return pred_obs, pred_rews
