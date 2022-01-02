import torch
from rich.progress import track
from torch import nn, optim
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_

from logging_callbacks import EnvModelWandb
from src.common import Params, get_env_configs
from src.env import get_env
from src.model import  RolloutStorage
from src.model.EnvModel import StochasticModel, NextFramePredictor
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


        split=params.minibatch//2

        # discard last state since prediction is on the next one
        input_states_batch = states_batch[:params.num_frames]
        target_states_batch = states_batch[params.num_frames+1]
        actions_batch = actions_batch[params.num_frames + 1]
        reward_batch = reward_batch[params.num_frames + 1]

        channel_size=params.obs_shape[0]

        input_states_batch = input_states_batch.resize(1, channel_size * params.num_frames, *params.obs_shape[1:])
        target_states_batch=target_states_batch.unsqueeze(dim=0)
        new_states_input = target_states_batch.float() / 255
        # call forward method
        imagined_frames, imagined_reward = env_model(
            input_states_batch, actions_batch, target=new_states_input
        )


        reward_loss = nn.CrossEntropyLoss()(imagined_reward, reward_batch)
        loss_reconstruct = nn.CrossEntropyLoss(reduction='none')(imagined_frames, target_states_batch.long())
        clip = torch.tensor(params.target_loss_clipping).to(params.device)
        loss_reconstruct = torch.max(loss_reconstruct, clip)
        loss_reconstruct = loss_reconstruct.mean() - params.target_loss_clipping

        loss_lstm = env_model.stochastic_model.get_lstm_loss()

        optimizer.zero_grad()
        loss = reward_loss + loss_reconstruct + loss_lstm
        loss.backward()

        mean_loss += loss.detach()

        logs = dict(
            reward_loss=reward_loss,
            image_loss=loss_reconstruct,
            imagined_state=imagined_frames[0].float(),
            actual_state=target_states_batch[0].float(),
            loss_lstm=loss_lstm
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
    env = get_env(env_config)

    obs_shape = env.reset().shape

    num_colors = len(params.color_index)

    rollout = RolloutStorage(
        num_steps=params.horizon * params.episodes + 10,
        size_minibatch=params.minibatch,
        obs_shape=obs_shape,
        num_actions=params.num_actions,
        num_agents=1
    )
    rollout.to(params.device)

    # ========================================
    #  init the env model
    # ========================================
    env_model=NextFramePredictor(params).to(params.device)


    optimizer = optim.RMSprop(
        env_model.parameters(), params.lr, eps=params.eps, alpha=params.alpha
    )

    env_model = env_model.to(params.device)
    env_model = env_model.train()

    # get logging step based on num of batches
    #num_batches = rollout.get_num_minibatches()
    #num_batches = int(num_batches * 0.01) + 1

    """wandb_callback = EnvModelWandb(
        train_log_step=num_batches,
        val_log_step=num_batches,
        project="env_model",
        model_config=params.__dict__,
        out_dir=params.WANDB_DIR,
        opts={},
        mode="disabled" if params.debug else "online",
    )"""

    for ep in track(range(params.epochs), description=f"Epochs"):
        # fill rollout storage with trajcetories
        collect_trajectories(params, env, rollout, obs_shape)
        rollout.to(params.device)

        # train for all the trajectories collected so far
        mean_loss = train_env_model(
            rollout, env_model, params, optimizer, None
        )
        rollout.after_update()

        # save result and log
        torch.save(env_model.state_dict(), "env_model.pt")
        #wandb_callback.on_epoch_end(loss=mean_loss, logs={}, model_path="env_model.pt")
