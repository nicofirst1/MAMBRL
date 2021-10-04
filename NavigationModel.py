from ray.rllib.models.tf import FullyConnectedNetwork, TFModelV2


class NavModel(TFModelV2):
    """Example of a keras custom model that just delegates to an fc-net."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(NavModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )
        self.model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        actions = self.model.forward(input_dict, state, seq_lens)
        return actions

    def value_function(self):
        return self.model.value_function()
