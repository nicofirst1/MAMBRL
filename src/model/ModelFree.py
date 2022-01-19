import math
from typing import Tuple

import torch
import torch.nn as nn
from torchvision.transforms import transforms

from src.common import FixedCategorical, init
from src.common.distributions import Categorical


class Flatten(nn.Module):
    @staticmethod
    def forward(x):
        return x.view(x.size(0), -1)

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

        #todo: base devono diventare feature extractions, ritornano solo features e occasionalemnte rnn
        if base == "resnet":
            base = ResNetBase
        elif base == "cnn":
            base = CNNBase
        else:
            raise NotImplementedError

        #todo: splittare gradienti per gruppo

        #todo: tieni buffer interno per RNN

        self.base = base(obs_shape, **base_kwargs)
        self.dist = Categorical(self.base.output_size, action_space)

    def get_modules(self):
        return self.base.get_modules()

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, masks)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a forward pass extracting the actor features and returning the values
        :param input:
        :return:
            action_logits: a torch tensor of size [batch_size, num_actions]
            values: the value from the value layer

        """
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        if actor_features.isnan().any():
            print()

        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        return self.base(inputs, rnn_hxs, masks)[0]

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

    def compute_action_entropy(self, frames: torch.Tensor):
        """evaluate_actions method.

        compute the actions logit, value and actions probability by passing
        the actual states (frames) to the ModelFree network. Then computes
        the entropy of the actions
        Parameters
        ----------
        frames : PyTorch Array
            a 4 dimensional tensor [batch_size, channels, width, height]

        Returns
        -------
        action_logit : Torch.Tensor [batch_size, num_actions]
            output of the ModelFree network before passing it to the softmax
        action_log_probs : torch.Tensor [batch_size, num_actions]
            log_probs of all the actions
        probs : Torch.Tensor [batch_size, num_actions]
            probability of actions given by the ModelFree network
        value : Torch.Tensor [batch_size,1]
            value of the state given by the ModelFree network
        entropy : Torch.Tensor [batch_size,1]
            value of the entropy given by the action with index equal to
            action_indx.
        """

        #todo: usala come evaluate actions, elimina categorical
        assert self.network_type != "critic", \
            "This network is set as critic, it cannot compute the action entropy"
        if self.network_type == "actor-critic":
            action_logit, value = self.forward(frames)
            action_probs = F.softmax(action_logit, dim=1)
            action_log_probs = F.log_softmax(action_logit, dim=1)
            entropy = -(action_probs * action_log_probs).sum(1).mean()

            return action_logit, action_log_probs, action_probs, value, entropy

        elif self.network_type == "actor":
            action_logit = self.forward(frames)
            action_probs = F.softmax(action_logit, dim=1)
            action_log_probs = F.log_softmax(action_logit, dim=1)
            entropy = -(action_probs * action_log_probs).sum(1).mean()

            return action_logit, action_log_probs, action_probs, entropy

        elif self.network_type == "critic":
            raise


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        #todo: remove gru from base
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
            has_zeros = ((masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu())

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

    def get_modules(self):
        return {}

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            hxs*= masks.unsqueeze(1)
            x, hxs = self.gru(x.unsqueeze(0), (hxs).unsqueeze(0))
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


class Conv2DModelFree(NNBase):
    """Conv2DModelFree class.

    Applies a 2D convolution over a frame
    """
    #todo: tieni solo convolution e feature extr
    def __init__(self, in_shape, num_actions, **kwargs):
        assert kwargs["num_frames"] == 0, "The parameter num_frames should be zero when using the 2D convolution"
        fc_layers = kwargs["fc_layers"]
        network_type = kwargs["network_type"]
        super(Conv2DModelFree, self).__init__(
            num_actions, features_out=fc_layers[-1], network_type=network_type)
        conv_layers = kwargs["conv_layers"]
        use_residual = kwargs["use_residual"]

        self.name = "Conv2DModelFree"
        shared_layers = OrderedDict()
        self.num_actions = num_actions
        self.in_shape = in_shape
        self.num_channels = in_shape[0]
        next_inp = None
        # =============================================================================
        # FEATURE EXTRACTOR SUBMODULE
        # =============================================================================
        for i, cnn in enumerate(conv_layers):
            if i == 0:
                shared_layers[network_type + "_conv_0"] = nn.Conv2d(
                    self.num_channels, cnn[0], kernel_size=cnn[1], stride=cnn[2])
                shared_layers[network_type + "_conv_0_activ"] = nn.LeakyReLU()
            else:
                shared_layers[network_type + "_conv_" + str(i)] = nn.Conv2d(
                    next_inp, cnn[0], kernel_size=cnn[1], stride=cnn[2])
                shared_layers[network_type + "_conv_" +
                              str(i) + "_activ"] = nn.LeakyReLU()
            next_inp = cnn[0]

        for layer in shared_layers:
            if layer == (network_type + "_conv_0"):
                fake_inp = torch.zeros(
                    [1, self.num_channels, *self.in_shape[1:]])
                fake_inp = shared_layers[layer](fake_inp)
            else:
                fake_inp = shared_layers[layer](fake_inp)
        # flatten layer
        next_inp = fake_inp.view(1, -1).size(1)

        # flatten the output starting from dim=1 by default
        shared_layers[network_type + "_flatten"] = nn.Flatten()
        shared_layers[network_type +
                      "_fc_0"] = nn.Linear(next_inp, fc_layers[0])
        shared_layers[network_type + "_fc_0_activ"] = nn.LeakyReLU()
        next_inp = fc_layers[0]
        self.shared_network = nn.Sequential(shared_layers)

        # =============================================================================
        # ACTOR AND CRITIC SUBNETS
        # =============================================================================
        if self.network_type == "actor-critic":
            actor_subnet = OrderedDict()
            critic_subnet = OrderedDict()
            for i, fc in enumerate(fc_layers[1:]):
                # Separate submodules for the actor and the critic
                actor_subnet["actor_fc_" + str(i)] = nn.Linear(next_inp, fc)
                critic_subnet["critic_fc_" +
                              str(i)] = nn.Linear(next_inp, fc)
                actor_subnet["actor_fc_" + str(i) + "_activ"] = nn.Tanh()
                critic_subnet["critic_fc_" +
                              str(i) + "_activ"] = nn.Tanh()
                next_inp = fc
            actor_subnet["actor_out"] = self.actor
            critic_subnet["critic_out"] = self.critic

            self.actor_network = nn.Sequential(actor_subnet)
            self.critic_network = nn.Sequential(critic_subnet)

        elif self.network_type == "actor":
            actor_subnet = OrderedDict()
            for i, fc in enumerate(fc_layers[1:]):
                # Separate submodules for the actor and the critic
                actor_subnet["actor_fc_" + str(i)] = nn.Linear(next_inp, fc)
                actor_subnet["actor_fc_" + str(i) + "_activ"] = nn.Tanh()
                next_inp = fc
            actor_subnet["actor_out"] = self.actor
            self.shared_network = nn.Sequential(shared_layers)
            self.actor_network = nn.Sequential(actor_subnet)

        elif self.network_type == "critic":
            critic_subnet = OrderedDict()
            for i, fc in enumerate(fc_layers[1:]):
                # Separate submodules for the actor and the critic
                critic_subnet["critic_fc_" +
                              str(i)] = nn.Linear(next_inp, fc)
                critic_subnet["critic_fc_" +
                              str(i) + "_activ"] = nn.Tanh()
                next_inp = fc
            critic_subnet["critic_out"] = self.critic
            self.shared_network = nn.Sequential(shared_layers)
            self.critic_network = nn.Sequential(critic_subnet)

    def to(self, device):
        if self.network_type == "actor_critic":
            self.critic = self.critic.to(device)
            self.actor = self.actor.to(device)
        else:
            self.shared_network = self.shared_network.to(device)
            if self.network_type == "actor":
                self.actor_network = self.actor_network.to(device)
            else:
                self.critic_network = self.critic_network.to(device)

        return self

    def forward(self, input):
        """forward method.

        Return the logit and the values of the Conv2DModelFree if the
        network_type is 'actor_critic'. Otherwise return only the value or
        only the logit
        Parameters
        ----------
        input : torch.Tensor
            [batch_size, num_channels, width, height]

        Returns
        -------
        logit : torch.Tensor
            [batch_size, num_actions]
        value : torch.Tensor
            [batch_size, value]

        """
        # last 2 layers shared layers requires a flattened input
        x = self.shared_network(input)

        if self.network_type == "actor-critic":
            action_logits = self.actor_network(x)
            value = self.critic_network(x)
            return action_logits, value

        elif self.network_type == "actor":
            action_logits = self.actor_network(x)
            return action_logits

        elif self.network_type == "critic":
            value = self.critic_network(x)
            return value


class CNNBase(NNBase):
    def __init__(self, input_shape, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent=recurrent, hidden_size=hidden_size, recurrent_input_size=hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x:
            nn.init.constant_(x, 0), nn.init.calculate_gain("relu"))

        num_inputs = input_shape[0]
        middle_shape = input_shape[1:]
        middle_shape = (middle_shape[0] // 4 - 1, middle_shape[1] // 4 - 1)
        middle_shape = (middle_shape[0] // 2 - 1, middle_shape[1] // 2 - 1)
        middle_shape = (middle_shape[0] - 2, middle_shape[1] - 2)

        self.features = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * middle_shape[0] * middle_shape[1], hidden_size)), nn.ReLU())

        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

        self.critic = init_(nn.Linear(hidden_size, 1))
        self.train()

    def get_modules(self):
        return dict(
            value=self.critic,
            features=self.features
        )

    def forward(self, inputs, rnn_hxs, masks):
        x = self.features(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic(x), x, rnn_hxs


class ResNetBase(NNBase):
    def __init__(self, input_shape, recurrent=False, hidden_size=512):
        super(ResNetBase, self).__init__(recurrent, hidden_size, hidden_size)

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

    def forward(self, inputs, rnn_hxs, masks):
        x = self.preprocess(inputs / 255.0)
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.hidden_layer(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.classifier(x), x, rnn_hxs
