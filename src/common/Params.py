import multiprocessing
import os
import uuid

import torch


class Params:
    unique_id = str(uuid.uuid1())[:8]

    #### DIRECTORIES ####
    WORKING_DIR = os.getcwd().split("src")[0]
    SRC_DIR = os.path.join(WORKING_DIR, "src")
    LOG_DIR = os.path.join(WORKING_DIR, "log_dir")
    RAY_DIR = os.path.join(LOG_DIR, "ray_results")
    EVAL_DIR = os.path.join(LOG_DIR, "eval")

    #### TRAINING ####
    debug = True
    device = torch.device("cuda")
    resize = True
    obs_shape = [3, 32, 32]
    num_workers = multiprocessing.cpu_count() - 1
    num_gpus = torch.cuda.device_count()
    framework = "torch"
    minibatch = 2
    epochs = 1

    #### ENVIRONMENT ####
    agents = 2
    landmarks = 2
    horizon = 21
    episodes = 3
    env_name = "collab_nav"
    model_name = f"{env_name}_model"
    obs_type = "image"  # or "states"
    num_frames = 3
    num_steps = horizon // num_frames
    full_rollout = False
    gray_scale = True

    #### EVALUATION ####
    log_step = 500
    checkpoint_freq = 50
    resume_training = False
    alternating = False
    max_checkpoint_keep = 10

    # Config Dict
    configs = {
        "ppo_clip_param": 0.1,
        "value_loss_coef": 1.0,
        "entropy_coef": 0.01,
        "max_grad_norm": 5
    }

    def __init__(self):
        if self.debug:
            self.device = torch.device("cpu")
            self.num_workers = 1
            self.num_gpus = 0
            torch.autograd.set_detect_anomaly(True)

        if self.gray_scale:
            self.obs_shape[0] = 1
