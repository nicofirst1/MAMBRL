import os
import uuid
import multiprocessing
import tensorflow as tf

class Params:
    unique_id = str(uuid.uuid1())[:8]

    #### DIRECTORIES ####
    WORKING_DIR = os.getcwd().split("src")[0]
    SRC_DIR     = os.path.join(WORKING_DIR, "src")
    LOG_DIR     = os.path.join(WORKING_DIR, "log_dir")
    RAY_DIR     = os.path.join(LOG_DIR, "ray_results")
    EVAL_DIR    = os.path.join(LOG_DIR, "eval")

    #### TRAINING ####
    debug       = False
    num_workers = multiprocessing.cpu_count()-1 if not debug else 1
    num_gpus    = len(tf.config.list_physical_devices('GPU')) if not debug else 0
    framework   = "tfe"

    #### ENVIRONMENT ####
    agents      = 2
    landmarks   = 3
    horizon     = 100
    episodes    = 5
    env_name    = "collab_nav"
    model_name  = f"{env_name}_model"

    #### EVALUATION ####
    log_step            = 500
    checkpoint_freq     = 50
    resume_training     = False
    alternating         = False
    max_checkpoint_keep = 10

    #### Config Dict
    configs={}

