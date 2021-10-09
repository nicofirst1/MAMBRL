import multiprocessing
import tensorflow as tf

class Params:
    debug = False

    #### TRAINING params
    num_cpus = multiprocessing.cpu_count() if not debug else 1
    num_gpus = len(tf.config.list_physical_devices('GPU')) if not debug else 0
    framework = "tfe"
    use_critic = True
    use_gae = True
    lambda_value = 1.0
    kl_coeff = 0.2
    rollout_fragment_length = 200
    train_batch_size = 4000
    sgd_minibatch_size = 128
    shuffle_sequences = True
    num_sgd_iter = 30
    lr = 5e-5
    lr_schedule = None
    vf_loss_coeff = 1.0
    entropy_coeff = 0.0
    entropy_coeff_schedule = None
    clip_param = 0.3
    vf_clip_param = 10.0
    grad_clip = None
    kl_target = 0.01
    batch_mode = "truncate_episodes"
    observation_filter = "NoFilter"

    #### ENV params
    agents = 2
    landmarks = 3
    horizon = 100
    episodes = 5
    env_name = "collab_nav"
    model_name = f"{env_name}_model"

    #### Config Dict
    configs={}

