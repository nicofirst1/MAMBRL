import torch
import torch.nn as nn
from torch.nn import Flatten
from torchvision.transforms import transforms

from src.common import FixedCategorical, init


def tan_init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def relu_init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def init_multi_layer(num_layers, activation, input_size, hidden_size, output_size):

    if num_layers==1:
        return nn.Sequential(
            nn.Linear(input_size,output_size)
        )

    module_list = [
        nn.Linear(input_size, hidden_size),
        activation(),
    ]

    for idx in range(num_layers - 1):
        module_list += [nn.Linear(hidden_size, hidden_size), activation()]

    module_list.append(
        nn.Linear(hidden_size, output_size)
    )


    return nn.Sequential(*module_list)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base, hidden_size, base_kwargs):
        super(Policy, self).__init__()

        if base == "resnet":
            base = ResNetBase
        elif base == "cnn":
            base = CNNBase
        else:
            raise NotImplementedError

        self.base = base(obs_shape, **base_kwargs)


        self.actions_layer = init_multi_layer(3, nn.Tanh,self.base.output_size, hidden_size,action_space)

        self.actions_layer.apply(tan_init_weights)

    def get_modules(self):

        modules = self.base.get_modules()
        modules['action_layer'] = self.actions_layer

        return modules

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, masks, recurrent_hidden_states, deterministic=False, full_log_prob=False, ):
        value, actor_features, rnn_hxs = self.base(inputs=inputs,
                                                   masks=masks,
                                                   rnn_hxs=recurrent_hidden_states)
        logits = self.actions_layer(actor_features)
        dist = FixedCategorical(logits=logits)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        if full_log_prob:
            action_log_probs = torch.log_softmax(logits, dim=-1)
        else:
            action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, masks, recurrent_hidden_states):
        return self.base(inputs, masks=masks,
                         rnn_hxs=recurrent_hidden_states)[0]

    def evaluate_actions(self, inputs, action, masks, recurrent_hidden_states, full_log_prob=False):
        value, actor_features, rnn_hxs = self.base(inputs, masks=masks,
                                                   rnn_hxs=recurrent_hidden_states)
        logits = self.actions_layer(actor_features)
        dist = FixedCategorical(logits=logits)

        if full_log_prob:
            action_log_probs = torch.log_softmax(logits, dim=-1)
        else:
            action_log_probs = dist.log_probs(action)

        min_real = torch.finfo(logits.dtype).min
        logits = torch.clamp(logits, min=min_real)
        p_log_p = logits * dist.probs
        dist_entropy = -p_log_p.sum(-1).mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, hidden_size, recurrent, recurrent_input_size, ):
        super(NNBase, self).__init__()
        self._hidden_size = hidden_size

        self._recurrent = recurrent
        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    def get_modules(self):
        return {}

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, input_shape, hidden_size=512, recurrent=False):
        super(CNNBase, self).__init__(hidden_size, recurrent=recurrent, recurrent_input_size=hidden_size)

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        num_inputs = input_shape[0]
        middle_shape = input_shape[1:]
        middle_shape = (middle_shape[0] // 4 - 1, middle_shape[1] // 4 - 1)
        middle_shape = (middle_shape[0] // 2 - 1, middle_shape[1] // 2 - 1)
        middle_shape = (middle_shape[0] - 2, middle_shape[1] - 2)

        self.features = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)),
            nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * middle_shape[0] * middle_shape[1], hidden_size)),
            nn.ReLU(),
        )

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.classifier = init_(nn.Linear(hidden_size, 1))

        self.train()

    def get_modules(self):
        return dict(
            value=self.classifier,
            features=self.features
        )

    def forward(self, inputs):
        x = self.features(inputs / 255.0)
        return self.classifier(x), x


class ResNetBase(NNBase):
    def __init__(self, input_shape, hidden_size=512, recurrent=False):
        super(ResNetBase, self).__init__(hidden_size, recurrent=recurrent, recurrent_input_size=hidden_size)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406] * 4, std=[0.229, 0.224, 0.225] * 4
                ),
            ]
        )
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # remove last linear layer
        self.features = (list(model.children())[:-1])

        # add initial convolution for stacked frames input
        num_inputs = input_shape[0]

        self.conv_0 = init_(
            nn.Conv2d(num_inputs, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        )
        self.features[0] = self.conv_0

        end = self.features[-1]
        # get up to the N+1 layer and discard the rest
        self.features = self.features[:7]
        # add average pool as last
        self.features[-1] = end
        self.features = torch.nn.Sequential(*self.features)



        self.hidden_layer=init_multi_layer(1,nn.ReLU,128,hidden_size,hidden_size)

        self.classifier= init_multi_layer(2, nn.ReLU, 64, hidden_size, 1)

        self.classifier.apply(relu_init_weights)
        self.hidden_layer.apply(relu_init_weights)

        self.train()

    def get_modules(self):
        return dict(
            value=self.classifier,
            hidden_layer=self.hidden_layer,
            features=self.features
        )

    def forward(self, inputs, masks, rnn_hxs):
        x = self.preprocess(inputs / 255.0)
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.hidden_layer(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        values = self.classifier(x)
        return values, x, rnn_hxs
