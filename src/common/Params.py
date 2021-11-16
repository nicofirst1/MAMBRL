import multiprocessing
import os
import uuid

import tensorflow as tf
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
    obs_shape = (3, 32, 32)
    num_workers = multiprocessing.cpu_count() - 1
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    framework = "torch"

    #### ENVIRONMENT ####
    agents = 2
    landmarks = 2
    horizon = 10
    episodes = 5
    env_name = "collab_nav"
    model_name = f"{env_name}_model"
    obs_type = "image"  # or "states"
    num_frames = 2
    num_steps = horizon // num_frames
    full_rollout = False

    #### EVALUATION ####
    log_step = 500
    checkpoint_freq = 50
    resume_training = False
    alternating = False
    max_checkpoint_keep = 10

    #### Config Dict
    configs = {}

    def __init__(self):
        if self.debug:
            self.device = torch.device("cpu")
            self.num_workers = 1
            self.num_gpus = 0
            torch.autograd.set_detect_anomaly(True)