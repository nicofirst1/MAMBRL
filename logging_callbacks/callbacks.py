import argparse
import os
from typing import Any, Dict, Optional, Union

import wandb
from src.common import Params


class WandbLogger:
    def __init__(
        self,
        opts: Union[argparse.ArgumentParser, Dict, str, None] = {},
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

        params = Params()
        out_dir = params.WANDB_DIR
        # create wandb dir if not existing
        if not os.path.isdir(out_dir):
            os.mkdir(out_dir)

        self.run=wandb.init(
            project=project,
            id=run_id,
            dir=out_dir,
            entity="mambrl",
            config=params.__dict__,
            mode="disabled" if params.debug else "online",
            **kwargs,
        )
        wandb.config.update(opts)
        self.params=params

    @staticmethod
    def log_to_wandb(metrics: Dict[str, Any], commit: bool = False, **kwargs):
        wandb.log(metrics, commit=commit, **kwargs)

    def wandb_close(self):
        """close method.

        it ends the current wandb run
        """
        wandb.finish()
