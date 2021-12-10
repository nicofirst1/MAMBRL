import torch
from rich.progress import track
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from logging_callbacks import EnvModelWandb
from src.common import Params, get_env_configs
from src.env import get_env
from src.model import EnvModel, RolloutStorage, target_to_pix
from src.train.train_utils import collect_trajectories


def train_env_model(rollouts, env_model, params, optimizer, callback_fn):

    # get data generation that splits rollout in batches
    data_generator = rollouts.recurrent_generator()

    criterion = nn.MSELoss()

    mean_loss = 0

    for batch_id, sample in enumerate(data_generator):
        (
            states_batch,
            actions_batch,
            _,
            reward_batch,
            _,
            _,
            _,
        ) = sample

        # discard last state since prediction is on the next one
        input_states_batch = states_batch[:-1]
        output_states_batch = states_batch[1:]
        actions_batch = actions_batch[:-1]
        reward_batch = reward_batch[:-1]

        # call forward method
        imagined_state, imagined_reward = env_model.full_pipeline(
            actions_batch, input_states_batch
        )

        reward_loss = (reward_batch.float() - imagined_reward).pow(2).mean()
        reward_loss = Variable(reward_loss, requires_grad=True)
        image_loss = criterion(imagined_state, output_states_batch)

        optimizer.zero_grad()
        loss = reward_loss + image_loss
        loss.backward()

        mean_loss += loss.detach()

        logs = dict(
            reward_loss=reward_loss,
            image_loss=image_loss,
            imagined_state=imagined_state[0].float(),
            actual_state=output_states_batch[0].float(),
        )

        callback_fn(logs=logs, loss=loss, batch_id=batch_id, is_training=True)

        clip_grad_norm_(env_model.parameters(), params.max_grad_norm)
        optimizer.step()

    return mean_loss / batch_id


if __name__ == "__main__":

    params = Params()

    # ========================================
    # get all the configuration parameters
    # ========================================

    params.agents = 1
    env_config = get_env_configs(params)
    env_config["mode"] = "rgb_array"
    env = get_env(env_config)

    obs_shape = env.reset().shape

    num_colors = len(params.color_index)

    t2p = target_to_pix(params.color_index, gray_scale=params.gray_scale)

    rollout = RolloutStorage(
        params.horizon * params.episodes,
        obs_shape,
        num_agents=params.agents,
        gamma=params.gamma,
        size_minibatch=params.minibatch,
    )
    rollout.to(params.device)

    # ========================================
    #  init the env model
    # ========================================

    env_model = EnvModel(
        obs_shape,
        reward_range=env.get_reward_range(),
        num_frames=params.num_frames,
        num_actions=params.num_actions,
        num_colors=num_colors,
        target2pix=t2p,
    )

    optimizer = optim.RMSprop(
        env_model.parameters(), params.lr, eps=params.eps, alpha=params.alpha
    )

    env_model = env_model.to(params.device)
    env_model = env_model.train()

    # get logging step based on num of batches
    num_batches = rollout.get_num_minibatches()
    num_batches = int(num_batches * 0.01) + 1

    wandb_callback = EnvModelWandb(
        train_log_step=num_batches,
        val_log_step=num_batches,
        project="env_model",
        model_config=params.__dict__,
        out_dir=params.WANDB_DIR,
        opts={},
        mode="disabled",  # if params.debug else "online",
    )

    for ep in track(range(params.epochs), description=f"Epochs"):
        # fill rollout storage with trajcetories
        collect_trajectories(params, env, rollout, obs_shape)
        rollout.to(params.device)

        # train for all the trajectories collected so far
        mean_loss = train_env_model(
            rollout, env_model, params, optimizer, wandb_callback.on_batch_end
        )
        rollout.after_update()

        # save result and log
        torch.save(env_model.state_dict(), "env_model.pt")
        wandb_callback.on_epoch_end(loss=mean_loss, logs={}, model_path="env_model.pt")
