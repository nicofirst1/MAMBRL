import multiprocessing
import os
import uuid

import torch


class Params:
    unique_id = str(uuid.uuid1())[:8]

    #### DIRECTORIES ####
    WORKING_DIR = os.getcwd().split("MAMBRL")[0]
    WORKING_DIR = os.path.join(WORKING_DIR,"MAMBRL")
    SRC_DIR = os.path.join(WORKING_DIR, "src")
    LOG_DIR = os.path.join(WORKING_DIR, "log_dir")
    RAY_DIR = os.path.join(LOG_DIR, "ray_results")
    EVAL_DIR = os.path.join(LOG_DIR, "eval")
    WANDB_DIR = os.path.join(LOG_DIR, "wandb")

    #### TRAINING ####
    debug = False
    use_wandb = False
    device = torch.device("cuda")
    resize = True
    obs_shape = [12, 96, 96]
    frame_shape = [3, 96, 96]
    num_workers = multiprocessing.cpu_count() - 1
    num_gpus = torch.cuda.device_count()
    framework = "torch"
    param_sharing=False
    experiment_name="model_free"

    ### ENV model
    stack_internal_states=False
    recurrent_state_size=64
    hidden_size=96
    compress_steps=2
    filter_double_steps=3
    hidden_layers=2
    bottleneck_bits=128
    latent_state_size=128
    dropout=0.15
    bottleneck_noise= 0.1
    latent_rnn_max_sampling= 0.5
    latent_use_max_probability= 0.8
    residual_dropout=0.5
    target_loss_clipping=0.03

    ### Optimizer
    lr = 3e-4
    eps = 1e-5
    alpha = 0.99
    max_grad_norm = 5

    ### Algo parameters
    gamma = 0.998
    ppo_clip_param = 0.1

    ### Loss
    value_loss_coef = 0.5
    entropy_coef = 0.01

    #### ENVIRONMENT ####
    agents = 1
    landmarks = 1
    epochs = 1000
    minibatch = 8
    episodes = 3
    horizon =  64
    env_name = "collab_nav"
    model_name = f"{env_name}_model"
    obs_type = "image"  # or "states"
    num_frames = 4
    rollout_len = 1
    batch_size = 3
    num_steps = horizon // num_frames
    full_rollout = False
    gray_scale = False
    num_actions = 5

    #### EVALUATION ####
    log_step = 500
    checkpoint_freq = 50
    restore=True
    resume_training = False
    alternating = False
    max_checkpoint_keep = 10

    # Config Dict
    configs = {
        "rollout_fragment_length": 50,
        # PPO parameter
        "lambda": 0.95,
        "gamma": 0.998,
        "clip_param": 0.2,
        "use_critic": True,
        "use_gae": True,
        "grad_clip": 5,
        "num_sgd_iter": 10,
    }

    color_index = [  # map index to RGB colors
        (0, 255, 0),  # green -> landmarks
        (0, 0, 255),  # blue -> agents
        (255, 255, 255),  # white -> background
    ]

    def __init__(self):
        if self.debug:
            self.device = torch.device("cpu")
            self.num_workers = 1
            self.num_gpus = 0
            torch.autograd.set_detect_anomaly(True)

        if self.gray_scale:
            self.frame_shape[0] = 1

        self.obs_shape[0] = self.frame_shape[0] * self.num_frames

        self.__initialize_dirs()

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
