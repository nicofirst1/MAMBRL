from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import transforms

from src.common import init


def tan_init_weights(m: nn):
    """tan_init_weights function.

    given a layer m, if m is a fully connected layer (nn.Linear) weights are
    initialized with Xavier_initialization
    https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_uniform_
    Parameters
    ----------
    m : torch.nn
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def relu_init_weights(m):
    """tan_init_weights function.

    given a layer m, if m is a fully connected layer (nn.Linear) weights are
    initialized with kaiming_initialization
    https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
    Parameters
    ----------
    m : torch.nn
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def init_multi_layer(num_layers, activation, input_size, hidden_size, output_size):
    """init_multi_layer function.

    create and return a nn.Sequential module with num_layers nn.Linear layers,
    each followed by the activation function passed
    Parameters
    ----------
    num_layers : int
    activation : torch.nn activation
    input_size : int
    hidden_size : int
    output_size : int

    Returns
    -------
    torch.nn.Sequential

    """
    if num_layers == 1:
        return nn.Sequential(
            nn.Linear(input_size, output_size)
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


class ModelFree(nn.Module):
    """ModelFree class.

    main class for the ModelFree model, all model class should inherit from
    it.
    """

    def __init__(self, base: str, base_kwargs):
        super(ModelFree, self).__init__()

        if base == "resnet":
            base = ResNetBase
        elif base == "cnn":
            base = Conv2DModelFree
        else:
            Exception("Base model not supported")

        self.base = base(**base_kwargs)

    def get_modules(self):
        return self.base.get_modules()

    def get_all_parameters(self):
        return self.base.get_all_parameters()

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def feature_extraction(self, inputs, masks):
        return self.base.feature_extractor(inputs, masks)

    def forward(self, inputs, masks) -> Tuple[torch.Tensor, torch.Tensor]:
        """forward method.

        Perform a forward pass extracting the actor features and returning
        the values
        Parameters
        ----------
        inputs : torch.Tensor
        Returns
        -------
            action_logits : a torch tensor of size [batch_size, num_actions]
            values : the value from the value layer

        """
        # TODO add proper value instead of None
        # return self.base.forward(inputs, None, masks)
        return self.base.forward(inputs, masks)

    def act(self, inputs, masks, deterministic=False):
        # normalize the input outside
        action_logit, value = self.base(inputs, masks)

        action_probs = F.softmax(action_logit, dim=1)

        if deterministic:
            action = action_probs.max(1)[1]
        else:
            action = action_probs.multinomial(1)

        log_actions_prob = F.log_softmax(action_logit, dim=1).squeeze()

        if value.shape[0] == 1:
            value = float(value)
            action = int(action)

        return value, action, log_actions_prob

    @staticmethod
    def get_action(action_logit, deterministic=False):
        action_probs = F.softmax(action_logit, dim=1)

        if deterministic:
            action = action_probs.max(1)[1]
        else:
            action = action_probs.multinomial(1)

        log_actions_prob = F.log_softmax(action_logit, dim=1).squeeze()

        return action, log_actions_prob


    def get_value(self, inputs, masks):
        return self.base(inputs, masks)[1]

    def evaluate_actions(self, inputs: torch.Tensor, masks):
        """evaluate_actions method.

        compute the actions logit, value and actions probability by passing
        the actual states (frames) to the ModelFree network. Then computes
        the entropy of the actions
        Parameters
        ----------
        inputs : PyTorch Array
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
            value of the entropy given by the action with index equal to action_indx.
        """

        action_logit, value = self.base(inputs, masks)
        action_probs = F.softmax(action_logit, dim=1)
        action_log_probs = F.log_softmax(action_logit, dim=1)
        entropy = -(action_probs * action_log_probs).sum(1).mean()

        return value, action_log_probs, entropy


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

            # Let's figure out which steps in the sequence have a zero for any
            # agent. We will always assume t=0 has a zero in it as that makes
            # the logic cleaner
            has_zeros = ((masks[1:] == 0.0).any(
                dim=-1).nonzero().squeeze().cpu())

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


class FeatureExtractor(NNBase):
    """CNNBase class.

    base module for the extraction of features composed of convolutional
    layers. The parameters and the number of convolutional layers are fully
    customizable
    """

    def __init__(self, input_shape, conv_layers, fc_layers, recurrent=False, hidden_size=512, num_frames=1, **kwargs):
        super(FeatureExtractor, self).__init__(recurrent=recurrent,
                                               hidden_size=hidden_size, recurrent_input_size=hidden_size,)
        self.in_shape = input_shape
        self.num_channels = input_shape[0]
        self.num_frames = num_frames

        next_inp = None
        feature_extractor_layers = OrderedDict()

        # if num_frames == 1 we build a 2DConv base, otherwise we build a 3Dconv base
        for i, cnn in enumerate(conv_layers):
            if i == 0:
                feature_extractor_layers["conv_0"] = nn.Conv2d(self.num_channels, cnn[0], kernel_size=cnn[1], stride=cnn[2])
                feature_extractor_layers["conv_0_activ"] = nn.LeakyReLU()
            else:
                feature_extractor_layers["conv_" + str(i)] = nn.Conv2d(next_inp, cnn[0], kernel_size=cnn[1], stride=cnn[2])
                feature_extractor_layers["conv_" + str(i) + "_activ"] = nn.LeakyReLU()
            next_inp = cnn[0]

        for layer in feature_extractor_layers:
            if layer == "conv_0":
                fake_inp = torch.zeros([1, self.num_channels, *self.in_shape[1:]])
                fake_inp = feature_extractor_layers[layer](fake_inp)
            else:
                fake_inp = feature_extractor_layers[layer](fake_inp)

        # flatten layer
        next_inp = fake_inp.view(1, -1).size(1)

        # flatten the output starting from dim=1 by default
        feature_extractor_layers["flatten"] = nn.Flatten()
        feature_extractor_layers["fc_0"] = nn.Linear(next_inp, fc_layers[0])
        feature_extractor_layers["fc_0_activ"] = nn.LeakyReLU()
        self.model = nn.Sequential(feature_extractor_layers)

    def forward(self, inputs, masks):
        return self.model(inputs)


class Conv2DModelFree(nn.Module):
    """Conv2DModelFree class.

    Applies a 2D convolution over a frame
    """

    def __init__(self, obs_shape, share_weights, action_space, conv_layers, fc_layers,
                 use_recurrent, use_residual, num_frames, base_hidden_size):
        #assert num_frames == 1, "The parameter num_frames should be one when using the 2D convolution"
        assert 0 < len(fc_layers) < 3, f"fc_layers should be a tuple of lists of 1 or 2 elements while it's {fc_layers}"
        assert 0 < len(conv_layers) < 3, f"conv_layers should be a tuple of lists of 1 or 2 elements while it's {conv_layers}"
        super(Conv2DModelFree, self).__init__()

        # self, obs_shape, action_space, base, hidden_size, share_weights, base_kwargs
        self.name = "Conv2DModelFree"
        self.num_actions = action_space
        self.share_weights = share_weights

        fc_layers = fc_layers
        conv_layers = conv_layers
        if len(conv_layers) == 1:
            conv_layers = (conv_layers[0], conv_layers[0])
        if len(fc_layers) == 1:
            fc_layers = (fc_layers[0], fc_layers[0])

        if self.share_weights:
            self.feature_extractor = FeatureExtractor(
                obs_shape, conv_layers[0], fc_layers[0], recurrent=use_recurrent,
                hidden_size=base_hidden_size, num_frames=num_frames
            )
        else:
            self.feature_extractor_actor = FeatureExtractor(
                obs_shape, conv_layers[1], fc_layers[1], recurrent=use_recurrent,
                hidden_size=base_hidden_size, num_frames=num_frames
            )
            self.feature_extractor_critic = FeatureExtractor(
                obs_shape, conv_layers[0], fc_layers[0], recurrent=use_recurrent,
                hidden_size=base_hidden_size, num_frames=num_frames
            )

        # =============================================================================
        # CRITIC SUBNETS
        # =============================================================================
        next_inp = fc_layers[0][0]
        critic_subnet = OrderedDict()
        for i, fc in enumerate(fc_layers[0][1:]):
            critic_subnet["critic_fc_" + str(i)] = nn.Linear(next_inp, fc)
            critic_subnet["critic_fc_" + str(i) + "_activ"] = nn.Tanh()
            next_inp = fc
        critic_subnet["critic_out"] = nn.Linear(next_inp, 1)

        # =============================================================================
        # ACTOR SUBNETS
        # =============================================================================
        if self.share_weights:
            next_inp = fc_layers[0][0]
        else:
            next_inp = fc_layers[1][0]
        actor_subnet = OrderedDict()
        for i, fc in enumerate(fc_layers[1][1:]):
            actor_subnet["actor_fc_" + str(i)] = nn.Linear(next_inp, fc)
            actor_subnet["actor_fc_" + str(i) + "_activ"] = nn.Tanh()
            next_inp = fc

        actor_subnet["actor_out"] = nn.Linear(next_inp, action_space)

        self.actor = nn.Sequential(actor_subnet)
        self.critic = nn.Sequential(critic_subnet)

    def get_modules(self):

        modules_dict = {}
        if self.share_weights:
            modules_dict['feature_extractor'] = self.feature_extractor.model
        else:
            modules_dict['feature_extractor_actor'] = self.feature_extractor_actor.model
            modules_dict['feature_extractor_critic'] = self.feature_extractor_critic.model

        modules_dict["actor"] = self.actor
        modules_dict["critic"] = self.critic

        return modules_dict

    def to(self, device):
        if self.share_weights:
            self.feature_extractor.to(device)
        else:
            self.feature_extractor_critic.to(device)
            self.feature_extractor_actor.to(device)

        self.critic.to(device)
        self.actor.to(device)

        return self

    def forward(self, inputs, masks):
        """forward method.

        Return the logit and the values of the Conv2DModelFree.
        Parameters
        ----------
        inputs : torch.Tensor
            [batch_size, num_channels, width, height]

        Returns
        -------
        logit : torch.Tensor
            [batch_size, num_actions]
        value : torch.Tensor
            [batch_size, value]

        """

        if self.share_weights:
            x = self.feature_extractor.forward(inputs, masks)
            return self.actor(x), self.critic(x)
        else:
            x = self.feature_extractor_critic.forward(inputs, masks)
            value = self.critic(x)

            x = self.feature_extractor_actor.forward(inputs, masks)
            action_logits = self.actor(x)

            return action_logits, value

    def get_actor_parameters(self):
        """get_actor_parameters method.

        returns all parameters relating to the actor network
        Returns
        -------
        list
            DESCRIPTION.

        """
        if self.share_weights:
            return [{'params': self.feature_extractor.parameters()},
                    {'params': self.actor.parameters()}]
        else:
            return [{'params': self.feature_extractor_actor.parameters()},
                    {'params': self.actor.parameters()}]

    def get_critic_parameters(self):
        """get_critic_parameters method.

        returns all parameters relating to the critic network
        Returns
        -------
        list
            DESCRIPTION.

        """
        if self.share_weights:
            return [{'params': self.feature_extractor.parameters()},
                    {'params': self.critic.parameters()}]
        else:
            return [{'params': self.feature_extractor_critic.parameters()},
                    {'params': self.critic.parameters()}]

    def get_all_parameters(self):
        """get_all_parameters method.

        returns all parameters relating to the network
        Returns
        -------
        list
            DESCRIPTION.

        """
        if self.share_weights:
            return [{'params': self.feature_extractor.parameters()},
                    {'params': self.critic.parameters()},
                    {'params': self.actor.parameters()}]
        else:
            return [{'params': self.feature_extractor_critic.parameters()},
                    {'params': self.feature_extractor_actor.parameters()},
                    {'params': self.critic.parameters()},
                    {'params': self.actor.parameters()}
                    ]

    def feature_extractor(self, inputs, masks):
        if self.share_weights:
            x = self.feature_extractor.forward(inputs, masks)
        else:
            x1 = self.feature_extractor_critic.forward(inputs, masks)

            x2 = self.feature_extractor_actor.forward(inputs, masks)
            x = torch.cat((x1, x2), dim=-1)

        return x


class ResNetBase(NNBase):
    """ResNetBase class.

    Pretrained feature extraction class based on the resnet18 architecture
    """

    def __init__(self, input_shape, recurrent=False, hidden_size=512):
        super(ResNetBase, self).__init__(recurrent, hidden_size, hidden_size)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406] * 4,
                    std=[0.229, 0.224, 0.225] * 4
                ),
            ]
        )

        def init_(m): return init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain("relu"),
        )

        model = torch.hub.load("pytorch/vision:v0.10.0",
                               "resnet18", pretrained=True)
        model = model.eval()
        for param in model.parameters():
            param.requires_grad = False

        # remove last linear layer
        self.features = (list(model.children())[:-1])

        # add initial convolution for stacked frames input
        num_inputs = input_shape[0]

        self.conv_0 = init_(
            nn.Conv2d(num_inputs, 64, kernel_size=(7, 7), stride=(2, 2),
                      padding=(3, 3))
        )
        self.features[0] = self.conv_0

        end = self.features[-1]
        # get up to the N+1 layer and discard the rest
        self.features = self.features[:7]
        # add average pool as last
        self.features[-1] = end
        self.features = torch.nn.Sequential(*self.features)

        self.hidden_layer = init_multi_layer(
            1, nn.ReLU, 128, hidden_size, hidden_size)

        self.classifier = init_multi_layer(2, nn.ReLU, 64, hidden_size, 1)

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
        x = self.preprocess(inputs)
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.hidden_layer(x)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.classifier(x), x, rnn_hxs
