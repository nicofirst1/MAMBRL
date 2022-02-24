import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from typing import Tuple, List
from src.common.utils import one_hot_encode
from src.trainer.Policies import MultimodalMAS
from src.common.Params import Params
from src.model.ModelFree import FeatureExtractor, ModelFree
from src.trainer.Policies import TrajCollectionPolicy
from src.model.EnvModel import NextFramePredictor
from src.model.RolloutEncoder import RolloutEncoder


class FullModel(nn.Module):
    """FullModel class.

    The FullModel is a model build from different blocks. In summary, it contains
    2 model for feature extraction. These features are then concatenated and
    used as input to an actor critic network.
    """

    def __init__(self,
                 mf_feature_extractor: FeatureExtractor,
                 mb_actor_model: ModelFree,
                 env_model: NextFramePredictor,
                 rollout_encoder: RolloutEncoder,
                 config: Params):
        """
        :param mf_feature_extractor: feature extractor class used for the model free
            part of the architecture
        :param mb_actor: Policy calss, it select an action used by the env_model 
        :param env_model: class used to predict future frames from the actual frame
            the actual actions picked by the mb_actor
        :param rollout_encoder: use the output of the env_model to encode
            the predicted frames
        :param config: class with all the configuration parameters
        """
        super(FullModel, self).__init__()
        # =============================================================================
        # GENERIC PARAMS
        # =============================================================================
        self.input_shape = config.frame_shape
        self.num_actions = config.num_actions
        self.device = config.device

        # =============================================================================
        # BLOCKS
        # =============================================================================
        mf_feature_extractor_config = config.get_mf_feature_extractor_configs()
        # FIXME: hardcoded
        mf_feature_extractor_config["conv_layers"] = mf_feature_extractor_config["conv_layers"][0]
        mf_feature_extractor_config["fc_layers"] = mf_feature_extractor_config["fc_layers"][0]
        self.mf_feature_extractor = mf_feature_extractor(input_shape=self.input_shape,
                                                         **mf_feature_extractor_config).to(self.device)

        # FIXME: add a get_env_model_config() function
        self.env_model = env_model(config).to(self.device)
        rollout_encoder_config = config.get_rollout_encoder_configs()
        self.rollout_encoder = rollout_encoder(
            **rollout_encoder_config).to(self.device)

        # actor who chooses the actions to be used by the env_model
        mb_actor_config = config.get_model_free_configs()
        self.mb_actor = mb_actor_model(**mb_actor_config).to(self.device)

        next_inp = None
        features_shape = self.get_features()

        critic_fc_layers = config.fm_fc_layers[0]
        actor_fc_layers = config.fm_fc_layers[1]
        # CRITIC
        critic = OrderedDict()
        for i, fc in enumerate(critic_fc_layers):
            if i == 0:
                critic["fm_critic_fc_0"] = nn.Linear(features_shape, fc)
                critic["fm_critic_fc_0_activ"] = nn.Tanh()
            else:
                critic["fm_critic_fc" + str(i)] = nn.Linear(next_inp, fc)
                critic["fm_critic_fc_" + str(i) + "_activ"] = nn.Tanh()
            next_inp = fc
        critic["fm_critic_output"] = nn.Linear(next_inp, 1)
        critic["fm_critic_output_activ"] = nn.Tanh()
        self.critic = nn.Sequential(critic).to(self.device)
        # ACTOR
        actor = OrderedDict()
        for i, fc in enumerate(actor_fc_layers):
            if i == 0:
                actor["fm_actor_fc_0"] = nn.Linear(features_shape, fc)
                actor["fm_actor_fc_0_activ"] = nn.Tanh()
            else:
                actor["fm_actor_fc" + str(i)] = nn.Linear(next_inp, fc)
                actor["fm_actor_fc_" + str(i) + "_activ"] = nn.Tanh()
            next_inp = fc
        actor["fm_actor_output"] = nn.Linear(next_inp, self.num_actions)
        actor["fm_actor_output_activ"] = nn.Tanh()
        self.actor = nn.Sequential(actor).to(self.device)

    def get_features(self):
        """compute the dimension of the concatenated features.

        the concatenated features are the features from the model free and
        env model
        """
        fake_input = torch.zeros(1, *self.input_shape).to(self.device)
        # FIXME: Hardcoded mask, need to fix
        mask = torch.ones(1)
        mf_features = self.mf_feature_extractor(fake_input, mask)
        fake_action = torch.randint(self.num_actions, (1,))
        fake_action = one_hot_encode(
            fake_action, self.num_actions).to(self.device)
        fake_action = fake_action.float()
        em_output, reward_pred, value_pred = self.env_model(
            fake_input, fake_action)
        # FIXME: Hardcoded reward since its shape returned from the
        # env_model is not suitable
        em_output = torch.argmax(em_output, dim=1).unsqueeze(dim=0).float()
        fake_reward = torch.zeros((*em_output.shape[:2], 1)).to(self.device)
        em_features = self.rollout_encoder(em_output, fake_reward)
        features = torch.cat((mf_features, em_features), dim=-1)

        return features.shape[-1]

    def to(self, device):
        self.mf_feature_extractor.to(device)
        self.env_model.to(device)
        self.rollout_encoder.to(device)
        self.critic.to(device)
        self.actor.to(device)

        return self

    def forward(self, inputs):
        """standard forward pass

        Parameters
        ----------
        inputs : Torch.Tensor
            [batch_size, channels, width, height]
        Returns
        -------
        action_logits : Torch.Tensor
            [batch_size, num_actions]
        value : Torch.Tensor
            [batch_size, 1]

        """
        # FIXME: Hardcoded mask, need to fix
        batch_size = inputs.shape[0]
        mask = torch.ones(batch_size, 1)
        mf_features = self.mf_feature_extractor(inputs, mask)
        _, action, _ = self.mb_actor.act(inputs, mask)

        action = one_hot_encode(
            action, self.num_actions).to(self.device).unsqueeze(0)
        action = action.float()
        em_output, reward_pred, value_pred = self.env_model(inputs, action)
        em_output = torch.argmax(em_output, dim=1).unsqueeze(dim=0).float()
        fake_reward = torch.zeros((*em_output.shape[:2], 1)).to(self.device)
        em_features = self.rollout_encoder(em_output, fake_reward)
        features = torch.cat((mf_features, em_features), dim=-1)
        action_logits = self.actor(features)
        value = self.critic(features)

        return action_logits, value


if __name__ == "__main__":
    configs = Params()
    model = FullModel(FeatureExtractor, ModelFree,
                      NextFramePredictor, RolloutEncoder, configs)
    fake_input = torch.zeros(1, *configs.frame_shape).to(configs.device)
    action_logits, value = model(fake_input)
