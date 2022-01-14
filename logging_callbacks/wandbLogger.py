import random
from typing import Any, Dict, Optional

import numpy as np
import wandb
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from logging_callbacks.callbacks import WandbLogger
from pytorchCnnVisualizations.src.misc_functions import apply_colormap_on_image


class EnvModelWandb(WandbLogger):
    def __init__(
            self,
            train_log_step: int,
            val_log_step: int,
            models,
            **kwargs,
    ):
        """
        Logs env model training onto wandb
        Args:
            train_log_step:
            val_log_step:
            out_dir:
            model_config:
            **kwargs:
        """

        super(EnvModelWandb, self).__init__(**kwargs)

        wandb.watch(models, log_freq=1000, log_graph=True,  log="all")

        self.train_log_step = train_log_step if train_log_step > 0 else 2
        self.val_log_step = val_log_step if val_log_step > 0 else 2

        self.epoch = 0

    def on_batch_end(
            self, logs: Dict[str, Any],  batch_id: int, is_training: bool = True
    ):

        flag = "training" if is_training else "validation"

        log_step = self.train_log_step

        if not is_training:
            log_step = self.val_log_step


        if batch_id% log_step!=0:
            return

        image_log_step = log_step * 10

        wandb_log = {
            f"loss/total": logs["loss_reward"] + logs["loss_reconstruct"]+ logs["loss_value"],
            f"loss/reward": logs["loss_reward"],
            f"loss/reconstruct": logs["loss_reconstruct"],
            f"loss/value": logs["loss_value"],
            f"epoch": self.epoch,
        }

        if "loss_lstm" in logs:
            wandb_log["loss/lstm"]=logs['loss_lstm']

        # image logging_callbacks
        if batch_id % image_log_step == 0:
            imagined_state=logs["imagined_state"]
            imagined_state=(imagined_state - imagined_state.min()) / (imagined_state.max() - imagined_state.min())
            imagined_state*=255
            img_log = {
                f"{flag}_imagined_state": wandb.Image(imagined_state),
                f"{flag}_actual_state": wandb.Image(logs["actual_state"]),
            }

            wandb_log.update(img_log)

        self.log_to_wandb(wandb_log, commit=True)

    def on_epoch_end(self, loss: float, logs: Dict[str, Any], model_path: str):

        self.epoch += 1


class PPOWandb(WandbLogger):
    def __init__(
            self,
            train_log_step: int,
            val_log_step: int,
            models: Dict[str, nn.Module],
            horizon: int,
            action_meaning: Dict[str, str],
            cams: Optional[list],
            **kwargs,
    ):
        """
        Logs env model training onto wandb
        Args:
            train_log_step:
            val_log_step:
            out_dir:
            model_config:
            **kwargs:
        """

        super(PPOWandb, self).__init__(**kwargs)

        for idx, mod in enumerate(models.values()):
            wandb.watch(mod, log_freq=1000, log_graph=True, idx=idx, log="all")

        self.train_log_step = train_log_step if train_log_step > 0 else 2
        self.val_log_step = val_log_step if val_log_step > 0 else 2
        self.horizon = horizon
        self.action_meaning = action_meaning
        self.epoch = 0

        self.log_behavior_step = 10
        self.log_heatmap_step = 100

        # Grad cam
        self.cams = cams

    def on_batch_end(self, logs: Dict[str, Any], batch_id: int, rollout):

        logs["epoch"] = batch_id

        if batch_id % self.log_behavior_step == 0:
            done_idx = (rollout.masks == 0).nonzero(as_tuple=True)[0].cpu()

            if len(done_idx) > 1:
                done_idx = done_idx[0]

            states = (
                rollout.states[:done_idx][:, -3:, :, :].cpu().numpy().astype(np.uint8)
            )

            actions = rollout.actions[:done_idx].squeeze().cpu().numpy()
            rewards = rollout.rewards[:done_idx].squeeze().cpu().numpy()

            states = write_rewards(states, rewards)

            logs["behaviour"] = wandb.Video(states, fps=16, format="gif")
            logs["hist/actions"] = actions
            logs["hist/rewards"] = rewards
            logs["mean_reward"] = rewards.mean()
            logs["episode_lenght"] = done_idx

        if batch_id % self.log_heatmap_step == 0 and len(self.cams) != 0:

            # map heatmap on image
            idx = random.choice(range(done_idx))
            img = rollout.states[idx]
            reprs = []
            for c in self.cams:
                cam = c.generate_cam(img.clone().unsqueeze(dim=0))
                reprs.append((c.name, cam))
            img = img[-3:]
            img = np.uint8(img.cpu().data.numpy())
            img = img.transpose(2, 1, 0)
            img = Image.fromarray(img).convert("RGB")

            for name, rep in reprs:
                heatmap, heatmap_on_image = apply_colormap_on_image(img, rep, "hsv")

                logs[f"cams/{name}"] = wandb.Image(heatmap_on_image)

                # save_gradient_images(np.array(heatmap_on_image), f"{name}_heatmap_on_image", file_dir="imgs")

        self.log_to_wandb(logs, commit=True)


def on_epoch_end(self, loss: float, logs: Dict[str, Any], model_path: str):
        self.epoch += 1


def write_rewards(states, rewards):
    """
    Write reward on state image
    :param states:
    :param rewards:
    :return:
    """
    states = states.transpose((0, 2, 3, 1))
    states = [Image.fromarray(states[i]) for i in range(states.shape[0])]
    draws = [ImageDraw.Draw(img) for img in states]
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 10)

    if rewards.size == 1:
        rewards = np.expand_dims(rewards, 0)

    for idx in range(rewards.size):
        rew = rewards[idx]
        draw = draws[idx]
        draw.rectangle(((0, 0), (160, 10)), fill="black")
        draw.text((0, 0), f"Rew {rew}", font=font, fill=(255, 255, 255))

    states = [np.asarray(state) for state in states]
    states = np.asarray(states)
    states = states.transpose((0, 3, 1, 2))
    return states


def preprocess_logs(learn_output, mamrbl):
    value_loss, action_loss, entropy, rollout, logs = learn_output

    # merge logs with agent id
    new_logs = {}
    for agent, values in logs.items():
        new_key = f"agents/{agent}"

        for k, v in values.items():
            new_logs[f"{new_key}_{k}"] = np.asarray(v).mean()

    logs = new_logs

    general_logs = {
        "loss/value_loss": value_loss,
        "loss/action_loss": action_loss,
        "loss/entropy_loss": entropy,
        "loss/total": value_loss + action_loss - entropy,
        "curriculum/guided_learning": mamrbl.ppo_wrapper.guided_learning_prob,
        "curriculum/reward": mamrbl.real_env.get_curriculum()[0][0],
        "curriculum/landmark": mamrbl.real_env.get_curriculum()[1][0],
        "curriculum/lr": mamrbl.ppo_wrapper.get_learning_rate(),
        "curriculum/entropy_coef": mamrbl.ppo_wrapper.ppo_agent.entropy_coef,
    }

    logs.update(general_logs)

    return logs, rollout
