import os
from typing import Any, Dict

import wandb

from logging_callbacks.callbacks import WandbLogger


class EnvModelWandb(WandbLogger):
    def __init__(
        self,
        train_log_step,
        val_log_step,
        out_dir,
        model_config,
        **kwargs,
    ):

        # create wandb dir if not existing
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        super(EnvModelWandb, self).__init__(dir=out_dir, config=model_config, **kwargs)

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

        model_artifact = wandb.Artifact(
            "env_model", type="model", metadata=dict(self.model_config)
        )

        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)
