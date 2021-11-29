import argparse
import json
import os
import pathlib
import re
import sys
from typing import Dict, Any, Union, Optional, NamedTuple
import torch
import wandb


class Callback:
    def on_train_begin(self):  # noqa: F821
        pass

    def on_train_end(self):
        pass

    def on_early_stopping(
            self,
            train_loss: float,
            train_logs: Dict[str, Any],
            epoch: int,
            test_loss: float = None,
            test_logs: Dict[str, Any] = None,
    ):
        pass

    def on_validation_begin(self, epoch: int):
        pass

    def on_validation_end(self, loss: float, logs: Dict[str, Any], epoch: int):
        pass

    def on_epoch_begin(self, epoch: int):
        pass

    def on_epoch_end(self, loss: float, logs: Dict[str, Any], epoch: int):
        pass

    def on_batch_end(
            self, logs: Dict[str, Any], loss: float, batch_id: int, is_training: bool = True
    ):
        pass


class ConsoleLogger(Callback):
    def __init__(self, print_train_loss=False, as_json=False):
        self.print_train_loss = print_train_loss
        self.as_json = as_json

    def aggregate_print(self, loss: float, logs: Dict[str, Any], mode: str, epoch: int):
        dump = dict(loss=loss)
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.items())
        dump.update(aggregated_metrics)

        if self.as_json:
            dump.update(dict(mode=mode, epoch=epoch))
            output_message = json.dumps(dump)
        else:
            output_message = ", ".join(sorted([f"{k}={v}" for k, v in dump.items()]))
            output_message = f"{mode}: epoch {epoch}, loss {loss}, " + output_message
        print(output_message, flush=True)

    def on_validation_end(self, loss: float, logs:  Dict[str, Any], epoch: int):
        self.aggregate_print(loss, logs, "test", epoch)

    def on_epoch_end(self, loss: float, logs:  Dict[str, Any], epoch: int):
        if self.print_train_loss:
            self.aggregate_print(loss, logs, "train", epoch)


class WandbLogger(Callback):
    def __init__(
            self,
            opts: Union[argparse.ArgumentParser, Dict, str, None] = None,
            project: Optional[str] = None,
            run_id: Optional[str] = None,
            **kwargs,
    ):
        # This callback logs to wandb the interaction as they are stored in the leader process.
        # When interactions are not aggregated in a multigpu run, each process will store
        # its own Dict[str, Any] object in logs. For now, we leave to the user handling this case by
        # subclassing WandbLogger and implementing a custom logic since we do not know a priori
        # what type of data are to be logged.
        self.opts = opts

        wandb.init(project=project, id=run_id, **kwargs)
        wandb.config.update(opts)

    @staticmethod
    def log_to_wandb(metrics: Dict[str, Any], commit: bool = False, **kwargs):
        wandb.log(metrics, commit=commit, **kwargs)




class Checkpoint(NamedTuple):
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    optimizer_scheduler_state_dict: Optional[Dict[str, Any]]


class CheckpointSaver(Callback):
    def __init__(
            self,
            checkpoint_path: Union[str, pathlib.Path],
            checkpoint_freq: int = 1,
            prefix: str = "",
            max_checkpoints: int = sys.maxsize,
    ):
        """Saves a checkpoint file for training.
        :param checkpoint_path:  path to checkpoint directory, will be created if not present
        :param checkpoint_freq:  Number of epochs for checkpoint saving
        :param prefix: Name of checkpoint file, will be {prefix}{current_epoch}.tar
        :param max_checkpoints: Max number of concurrent checkpoint files in the directory.
        """
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        self.checkpoint_freq = checkpoint_freq
        self.prefix = prefix
        self.max_checkpoints = max_checkpoints
        self.epoch_counter = 0

    def on_epoch_end(self, loss: float, logs: Dict[str, Any], epoch: int):
        self.epoch_counter = epoch
        if self.checkpoint_freq > 0 and (epoch % self.checkpoint_freq == 0):
            filename = f"{self.prefix}_{epoch}" if self.prefix else str(epoch)
            self.save_checkpoint(filename=filename)

    def on_train_end(self):
        self.save_checkpoint(
            filename=f"{self.prefix}_final" if self.prefix else "final"
        )

    def save_checkpoint(self, filename: str):
        """
        Saves the game, agents, and optimizer states to the checkpointing path under `<number_of_epochs>.tar` name
        """
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)
        if len(self.get_checkpoint_files()) > self.max_checkpoints:
            self.remove_oldest_checkpoint()
        path = self.checkpoint_path / f"{filename}.tar"
        torch.save(self.get_checkpoint(), path)

    def get_checkpoint(self):
        optimizer_schedule_state_dict = None
        if self.trainer.optimizer_scheduler:
            optimizer_schedule_state_dict = (
                self.trainer.optimizer_scheduler.state_dict()
            )
        if self.trainer.distributed_context.is_distributed:
            game = self.trainer.game.module
        else:
            game = self.trainer.game
        return Checkpoint(
            epoch=self.epoch_counter,
            model_state_dict=game.state_dict(),
            optimizer_state_dict=self.trainer.optimizer.state_dict(),
            optimizer_scheduler_state_dict=optimizer_schedule_state_dict,
        )

    def get_checkpoint_files(self):
        """
        Return a list of the files in the checkpoint dir
        """
        return [name for name in os.listdir(self.checkpoint_path) if ".tar" in name]

    @staticmethod
    def natural_sort(to_sort):
        """
        Sort a list of files naturally
        E.g. [file1,file4,file32,file2] -> [file1,file2,file4,file32]
        """
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
        return sorted(to_sort, key=alphanum_key)

    def remove_oldest_checkpoint(self):
        """
        Remove the oldest checkpoint from the dir
        """
        checkpoints = self.natural_sort(self.get_checkpoint_files())
        os.remove(os.path.join(self.checkpoint_path, checkpoints[0]))
