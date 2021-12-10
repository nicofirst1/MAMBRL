import argparse
import json
import os
import pathlib
import re
import sys
from typing import Any, Dict, NamedTuple, Optional, Union

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

