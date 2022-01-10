import random
from typing import Any, Dict, Optional

import numpy as np
import wandb
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from logging_callbacks.callbacks import WandbLogger
from pytorchCnnVisualizations.src.misc_functions import apply_colormap_on_image, save_gradient_images


class EnvModelWandb(WandbLogger):
    def __init__(
            self,
            train_log_step: int,
            val_log_step: int,
            out_dir: str,
            model_config,
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

        self.train_log_step = train_log_step if train_log_step > 0 else 2
        self.val_log_step = val_log_step if val_log_step > 0 else 2
        self.model_config = model_config
        self.out_dir = out_dir

        self.epoch = 0

    def on_batch_end(
            self, logs: Dict[str, Any], loss: float, batch_id: int, is_training: bool = True
    ):

        flag = "training" if is_training else "validation"

        log_step = self.train_log_step

        if not is_training:
            log_step = self.val_log_step

        image_log_step = log_step * 10

        wandb_log = {
            f"{flag}_loss": loss,
            f"{flag}_reward_loss": logs["reward_loss"],
            f"{flag}_image_loss": logs["image_loss"],
            f"{flag}_epoch": self.epoch,
        }

        # image logging_callbacks
        if batch_id % image_log_step == 0:
            img_log = {
                f"{flag}_imagined_state": wandb.Image(logs["imagined_state"]),
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

        wandb.watch(models)

        self.train_log_step = train_log_step if train_log_step > 0 else 2
        self.val_log_step = val_log_step if val_log_step > 0 else 2
        self.horizon = horizon
        self.action_meaning = action_meaning
        self.epoch = 0

        self.log_behavior_step = 5
        self.log_heatmap_step = 10

        # Grad cam
        self.cams = cams

    def on_batch_end(self, logs: Dict[str, Any], batch_id: int, rollout):

        #logs = {k: sum(v) / len(v) for k, v in logs.items()}

        logs["epoch"] = batch_id

        if batch_id % self.log_behavior_step == 0:
            done_idx = (rollout.masks == 0).nonzero(as_tuple=True)[0]

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

                #save_gradient_images(np.array(heatmap_on_image), f"{name}_heatmap_on_image", file_dir="imgs")

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
    for idx in range(len(rewards)):
        rew = rewards[idx]
        draw = draws[idx]
        draw.rectangle(((0, 00), (160, 10)), fill="black")
        draw.text((0, 0), f"Rew {rew}", font=font, fill=(255, 255, 255))

    states = [np.asarray(state) for state in states]
    states = np.asarray(states)
    states = states.transpose((0, 3, 1, 2))
    return states
