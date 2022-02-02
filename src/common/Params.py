import argparse
import inspect
import os
import uuid
import numpy as np

import torch


class Params:
    unique_id = str(uuid.uuid1())[:8]

    # =============================================================================
    # DIRECTORIES
    # =============================================================================
    WORKING_DIR = os.getcwd().split("MAMBRL")[0]
    WORKING_DIR = os.path.join(WORKING_DIR, "MAMBRL")
    SRC_DIR = os.path.join(WORKING_DIR, "src")
    LOG_DIR = os.path.join(WORKING_DIR, "log_dir")
    EVAL_DIR = os.path.join(LOG_DIR, "eval")
    WANDB_DIR = os.path.join(LOG_DIR, "wandb")
    TENSORBOARD_DIR = os.path.join(WORKING_DIR, "tensorboard")
    MODEL_FREE_LOG_DIR = os.path.join(LOG_DIR, "model_free_log")
    MODEL_FREE_LOGGER_FILE = os.path.join(MODEL_FREE_LOG_DIR, "model_free_log.log")

    # =============================================================================
    # TRAINING
    # =============================================================================
    debug = False
    use_wandb = True
    device = torch.device("cuda")
    resize = True
    frame_shape = [3, 32, 32]  # [3, 96, 96]  # [3, 600, 600]
    # TODO: add description
    guided_learning_prob = 0.0
    epochs = 5000
    # number of learning iterations that the algorithm does on the same batch
    # of trajectories (trajectories are shuffled at each iteration)
    batch_epochs = 3
    # number of elements on which the algorithm performs a learning step
    minibatch = 32  # 64
    batch_size = 100
    framework = "torch"

    # =============================================================================
    # SCHEDULERS
    # =============================================================================
    use_lr_sched=True

    # =============================================================================
    # MULTIAGENT
    # =============================================================================
    param_sharing = False

    # =============================================================================
    #  OPTIMIZER
    # =============================================================================
    lr = 3e-4
    alpha = 0.99
    max_grad_norm = 5
    eps = 1e-5

    # =============================================================================
    #  ENV MODEL
    # =============================================================================
    stack_internal_states = True
    recurrent_state_size = 64
    hidden_size = 96
    compress_steps = 5
    filter_double_steps = 3
    hidden_layers = 2
    bottleneck_bits = 128
    latent_state_size = 128
    dropout = 0.15
    bottleneck_noise = 0.1
    latent_rnn_max_sampling = 0.5
    latent_use_max_probability = 0.8
    residual_dropout = 0.5
    target_loss_clipping = 0.03
    scheduled_sampling_decay_steps = 3000
    input_noise = 0.05
    use_stochastic_model = True
    clip_grad_norm = 1.0
    rollout_len = 10
    save_models = True

    # =============================================================================
    # ALGO PARAMETERS
    # =============================================================================
    gamma = 0.998
    ppo_clip_param = 0.1
    clip_value_loss = False
    # Clip param for the value function. Note that this is sensitive to the
    # scale of the rewards. If your expected V is large, increase this.
    vf_clip_param = 10.0
    # Loss
    # Coefficient of the value function loss. IMPORTANT: you must tune this if
    # you set actor and critic networks share weights (from rllib ppo
    # implementation)
    value_loss_coef = 1  # [0.5, 1]
    entropy_coef = 0.01  # [0.5, 0.1]

    # =============================================================================
    # MODEL FREE NETWORK PARAMETERS
    # =============================================================================
    # if to use the same architecture for the actor-critic
    share_weights = False
    # conv_layers is a tuple of 1 or 2 elements. If len(conv_layers)==1 we use
    # the same structure for both actor and critic network, otherwise the first
    # element are the layers of the critic and the second are the layers of
    # the actor. if share_weights=True we use the first element as shared architecture.
    # A single conv layer is a 3-ple (Channel_out, kernel_size, stride)
    # Conv3D kernel_size and stride can be int or a 3-ple
    # NOTE: for a single element tuple use the comma, e.g conv_layers = (list1,)

    # conv_layers = ([(64, (2, 3, 3), 1), (64, (1, 3, 3), 1), (32, 2, 1)],)

    # Conv2D kernel_size and stride can be int or a 2-ple
    conv_layers = ([(64, 4, 2), (32, 2, 2), (32, 2, 2)],
                   [(32, 4, 2), (32, 3, 2)])

    # same as the conv_layers
    fc_layers = ([128, 64, 32], [64, 32])
    # use recurrent neural network
    use_recurrent = False
    # recurrent_layers
    use_residual = False

    base = "cnn"  # [ cnn , resnet ]
    base_hidden_size = 64

    # =============================================================================
    # ENVIRONMENT
    # =============================================================================
    agents = 1
    landmarks = 2
    step_reward = -0.01
    landmark_reward = 2
    episodes = 3   # 3
    horizon = 100    # 100
    landmarks_positions = np.array([[0.0, -1.0], [0.0, 1.0]])  # None
    agents_positions = None  # np.array([[0.0, 0.0]])
    env_name = "collab_nav"
    model_name = f"{env_name}_model"
    obs_type = "image"  # or "states"
    num_frames = 1
    num_steps = horizon // num_frames
    gray_scale = False
    normalize_reward = True
    world_max_size = 3
    visible = False
    max_landmark_counter = 4
    landmark_penalty = -0.01  # -0.01   # -1
    border_penalty = -0.1
    # 0 don't move, 1 left, 2 right,  3 down, 4 top
    num_actions = 5

    #### ENVIRONMENT ENTITIES ####
    agent_size = 0.1
    landmark_size = 0.3

    #### EVALUATION ####
    log_step = 500
    checkpoint_freq = 50
    restore = True
    resume_training = False
    max_checkpoint_keep = 10

    color_index = [  # map index to RGB colors
        (0, 255, 0),  # green -> landmarks
        (0, 0, 255),  # blue -> agents
        (255, 255, 255),  # white -> background
    ]

    action_meanings = {0: "stop", 1: "up", 2: "down", 3: "right", 4: "left"}

    def __init__(self):
        self.__initialize_dirs()
        self.__parse_args()

        if self.base=="resnet":
            self.share_weights=True

        if self.gray_scale:
            self.frame_shape[0] = 1

        if not self.debug:
            torch.backends.cudnn.benchmark = True

        self.obs_shape = (
            self.frame_shape[0] * self.num_frames, *self.frame_shape[1:])

    def __initialize_dirs(self):
        """
        Initialize all the directories  listed above
        :return:
        """
        variables = [
            attr
            for attr in dir(self)
            if not callable(getattr(self, attr)) and not attr.startswith("__")
        ]
        for var in variables:
            if var.lower().endswith("dir"):
                path = getattr(self, var)
                if not os.path.exists(path):
                    print(f"Mkdir {path}")
                    os.makedirs(path)
            # change values based on argparse

    def __parse_args(self):
        """
        Use argparse to change the default values in the param class
        """

        att = self.__get_attributes()

        """Create the parser to capture CLI arguments."""
        parser = argparse.ArgumentParser()

        # for every attribute add an arg instance
        for k, v in att.items():
            if isinstance(v, bool):
                parser.add_argument(
                    "-" + k.lower(),
                    action="store_true",
                    default=v,
                )
            else:
                parser.add_argument(
                    "--" + k.lower(),
                    type=type(v),
                    default=v,
                )

        args, unk = parser.parse_known_args()
        for k, v in vars(args).items():
            self.__setattr__(k, v)

    def __get_attributes(self):
        """
        Get a dictionary for every attribute that does not have "filter_str" in it
        :return:
        """

        # get every attribute
        attributes = inspect.getmembers(self)
        # filter based on double underscore
        filter_str = "__"
        attributes = [elem for elem in attributes if filter_str not in elem[0]]
        # convert to dict
        attributes = dict(attributes)

        return attributes

    def get_env_configs(self):
        env_config = dict(
            horizon=self.horizon,
            continuous_actions=False,
            gray_scale=self.gray_scale,
            frame_shape=self.frame_shape,
            visible=self.visible,
            scenario_kwargs=dict(
                step_reward=self.step_reward,
                landmark_reward=self.landmark_reward,
                landmark_penalty=self.landmark_penalty,
                border_penalty=self.border_penalty,
                num_agents=self.agents,
                num_landmarks=self.landmarks,
                max_size=self.world_max_size,
            ),
        )

        return env_config

    def get_ppo_configs(self):
        ppo_configs = dict(
            rollout_fragment_length=self.rollout_len,
            gamma=self.gamma,
            clip_param=self.ppo_clip_param,
            grad_clip=self.clip_grad_norm,
        )
        return ppo_configs

    def get_model_free_configs(self):
        model_config = dict(
            base=self.base,
            base_kwargs=dict(
                obs_shape=self.obs_shape,
                share_weights=self.share_weights,
                action_space=self.num_actions,
                conv_layers=self.conv_layers,
                fc_layers=self.fc_layers,
                use_recurrent=self.use_recurrent,
                use_residual=self.use_residual,
                num_frames=self.num_frames,
                base_hidden_size=self.base_hidden_size
            )
        )
        return model_config

    def get_env_wrapper_configs(self):

        return dict(
            frame_shape=self.frame_shape,
            num_stacked_frames=self.num_frames,
            device=self.device,
            gamma=self.gamma,
        )
