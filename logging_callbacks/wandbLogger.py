from typing import Any, Dict, Optional

import numpy as np
import torch
import torchvision
import wandb
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from logging_callbacks.callbacks import WandbLogger
from pytorchCnnVisualizations.src.misc_functions import apply_colormap_on_image
from src.agent.RolloutStorage import RolloutStorage
from src.common import Params

params = Params()


class EnvModelWandb(WandbLogger):
    def __init__(
            self,
            train_log_step: int,
            val_log_step: int,
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

        self.epoch = 0



    def on_batch_end(
            self, logs: Dict[str, Any], batch_id: int, is_training: bool = True
    ):

        flag = "training" if is_training else "validation"

        log_step = self.train_log_step

        if not is_training:
            log_step = self.val_log_step

        if batch_id % log_step != 0:
            return

        image_log_step = log_step * 10

        wandb_log = {
            f"loss/total": logs["loss_reward"] + logs["loss_reconstruct"] + logs["loss_value"],
            f"loss/reward": logs["loss_reward"],
            f"loss/reconstruct": logs["loss_reconstruct"],
            f"loss/value": logs["loss_value"],
            f"curriculum/value": logs["epsilon"],
            f"epoch": self.epoch,
        }

        if "loss_lstm" in logs:
            wandb_log["loss/lstm"] = logs['loss_lstm']

        # image logging_callbacks
        if batch_id % image_log_step == 0:
            imagined_state = logs["imagined_state"]
            actual_state = logs["actual_state"]

            imagined_state = torch.stack(imagined_state)
            actual_state = torch.stack(actual_state)

            # bring imagined state in range 0 255
            imagined_state = (imagined_state - imagined_state.min()) / \
                             (imagined_state.max() - imagined_state.min())
            #imagined_state *= 255

            diff = abs(imagined_state - actual_state)

            fps = 5
            img_log = {
                f"imagined_state": wandb.Video(imagined_state, fps=fps, format="gif"),
                f"actual_state": wandb.Video(actual_state, fps=fps, format="gif"),
                f"diff": wandb.Video(diff, fps=fps, format="gif"),
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

        # done_idx = (rollout.masks == 0).nonzero(as_tuple=True)[0].cpu()
        #
        # if len(done_idx) > 1:
        #     done_idx = done_idx[0]

        states = (
            rollout.states[:rollout.step][:, -3:, :, :].cpu().numpy().astype(np.uint8)
        )

        actions = rollout.actions[:rollout.step].squeeze().cpu().numpy()
        rewards = rollout.rewards[:rollout.step].squeeze().cpu().numpy()

        grids = write_infos(states, rollout, self.params.action_meanings)

        logs["behaviour"] = wandb.Video(states, fps=16, format="gif")
        logs["behaviour_info"] = wandb.Video(grids, fps=10, format="gif")
        logs["hist/actions"] = actions
        logs["hist/rewards"] = rewards
        logs["mean_reward"] = rewards.mean()
        logs["episode_length"] = rollout.step

        if batch_id % self.log_heatmap_step == 0 and len(self.cams) != 0:

            # map heatmap on image
            # idx = random.choice(range(done_idx))
            idx = rollout.step
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
                heatmap, heatmap_on_image = apply_colormap_on_image(
                    img, rep, "hsv")

                logs[f"cams/{name}"] = wandb.Image(heatmap_on_image)

                # save_gradient_images(np.array(heatmap_on_image), f"{name}_heatmap_on_image", file_dir="imgs")

        self.log_to_wandb(logs, commit=True)


class FullWandb(WandbLogger):
    def __init__(
            self,
            train_log_step: int,
            val_log_step: int,
            horizon: int,
            action_meaning: Dict[str, str],
            cams: Optional[list],
            models: Dict[str, nn.Module] = {},
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

        super(FullWandb, self).__init__(**kwargs)

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

        states = (
                rollout.states[:rollout.step][:, -3:, :, :].cpu().numpy().astype(np.uint8) * 255.0
        )

        actions = rollout.actions[:rollout.step].squeeze().cpu().numpy()
        rewards = rollout.rewards[:rollout.step].squeeze().cpu().numpy()

        grids = write_infos(states, rollout, self.params.action_meanings)

        logs["behaviour"] = wandb.Video(states, fps=16, format="gif")
        logs["behaviour_info"] = wandb.Video(grids, fps=10, format="gif")
        logs["hist/actions"] = actions
        logs["hist/rewards"] = rewards
        logs["mean_reward"] = rewards.mean()
        logs["episode_length"] = rollout.step

        if batch_id % self.log_heatmap_step == 0 and len(self.cams) != 0:

            # map heatmap on image
            # idx = random.choice(range(done_idx))
            idx = rollout.step
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
                heatmap, heatmap_on_image = apply_colormap_on_image(
                    img, rep, "hsv")

                logs[f"cams/{name}"] = wandb.Image(heatmap_on_image)
            imagined_state = logs["imagined_state"]
            actual_state = logs["actual_state"]

            imagined_state = torch.stack(imagined_state)
            actual_state = torch.stack(actual_state)

            # bring imageine state in range 0 255
            imagined_state = (imagined_state - imagined_state.min()) / \
                             (imagined_state.max() - imagined_state.min())
            imagined_state *= 255

            diff = abs(imagined_state - actual_state)

            fps = 5
            img_log = {
                f"imagined_state": wandb.Video(imagined_state, fps=fps, format="gif"),
                f"actual_state": wandb.Video(actual_state, fps=fps, format="gif"),
                f"diff": wandb.Video(diff, fps=fps, format="gif"),
            }

            logs.update(img_log)

                # save_gradient_images(np.array(heatmap_on_image), f"{name}_heatmap_on_image", file_dir="imgs")

        self.log_to_wandb(logs, commit=True)


def on_epoch_end(self, loss: float, logs: Dict[str, Any], model_path: str):
    self.epoch += 1


def write_infos(states, rollout: RolloutStorage, action_meaning: Dict):
    """
    Write reward on state image
    :param states:
    :param rewards:
    :return:
    """

    font_size = 10

    img_size = 128
    batch_size = states.shape[0]
    state_shape = states.shape[-1]
    img_size = state_shape if state_shape > img_size else img_size

    info_img = np.zeros((batch_size, img_size, img_size, 3)).astype(np.uint8)

    info_img = [Image.fromarray(info_img[i]) for i in range(batch_size)]
    draws = [ImageDraw.Draw(img) for img in info_img]

    font = ImageFont.load_default()
    #font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", font_size)

    # if rewards.size == 1:
    #     rewards = np.expand_dims(rewards, 0)

    for idx in range(batch_size):
        rew = float(rollout.rewards[idx].squeeze().cpu())
        act = int(rollout.actions[idx].squeeze().cpu())
        ret = float(rollout.returns[idx].squeeze().cpu())
        val = float(rollout.value_preds[idx].squeeze().cpu())

        rew = f"{rew:.3f}"
        ret = f"{ret:.3f}"
        val = f"{val:.3f}"

        act_mean = action_meaning[act]

        # transform log prob into prob
        log_prob = rollout.action_log_probs[idx].squeeze().cpu().numpy()
        log_prob = np.exp(log_prob)
        log_prob = [f"{p:.3f}, " for p in log_prob]

        draw = draws[idx]

        draw.text((0, 0), f"Return: {ret}", font=font, fill=(255, 255, 255))
        draw.text((0, 20), f"Value: {val}", font=font, fill=(255, 255, 255))
        draw.text((0, 40), f"Rew: {rew}", font=font, fill=(255, 255, 255))
        draw.text((0, 60), f"Action: {act} ({act_mean})", font=font, fill=(255, 255, 255))
        draw.text((0, 80), f"Prob: [{''.join(log_prob[:2])}", font=font, fill=(255, 255, 255))
        draw.text((0, 90), f"{''.join(log_prob[2:])}]", font=font, fill=(255, 255, 255))

    # info_img[0].show()
    info_img = [np.asarray(img) for img in info_img]
    info_img = np.asarray(info_img)
    info_img = info_img.transpose((0, 3, 1, 2))

    if state_shape < img_size:
        diff = img_size - state_shape
        pad_l = diff // 2
        pad_r = pad_l
        if diff % 2 != 0:
            pad_r += 1

        states = np.pad(states, ((0, 0), (0, 0), (pad_l, pad_r), (pad_l, pad_r)))

    info_img = torch.as_tensor(info_img)
    states = torch.as_tensor(states)

    grids = []
    for idx in range(batch_size):
        grids.append(
            torchvision.utils.make_grid(
                torch.stack([states[idx], info_img[idx]])
            )
        )

    grids = np.stack(grids)
    # grids = grids.transpose((0, 2, 3, 1))

    # Image.fromarray(np.asarray(grids[1])).show()
    return grids


def preprocess_logs(learn_output, model_free):
    value_loss, action_loss, entropy, logs = learn_output

    # merge logs with agent id
    new_logs = {}
    for agent, values in logs.items():
        new_key = f"agents/{agent}"

        if isinstance(values,list):
            # we have a list of dict with the same values, convert to a dict of list
            values={key: [i[key] for i in values] for key in values[0]}
            values= {k: [x for sub in v for x in sub] for k, v in values.items()}

        for k, v in values.items():
            new_logs[f"{new_key}_{k}"] = np.asarray(v).mean()

    logs = new_logs

    strat = params.get_descriptive_strategy()
    reward_step_strategy, reward_collision_strategy, \
        landmark_reset_strategy, landmark_collision_strategy = model_free.cur_env.get_current_strategy()

    tbl = wandb.Table(columns=["list", "current strategy", "description"])

    tbl.add_data("reward_step", reward_step_strategy, strat["reward_step_strategy"][reward_step_strategy])
    tbl.add_data("reward_collision", reward_collision_strategy, strat["reward_collision_strategy"][reward_collision_strategy])
    tbl.add_data("landmark_reset", landmark_reset_strategy, strat["landmark_reset_strategy"][landmark_reset_strategy])
    tbl.add_data("landmark_collision", landmark_collision_strategy, strat["landmark_collision_strategy"][landmark_collision_strategy])

    general_logs = {
        "loss/value_loss": value_loss,
        "loss/action_loss": action_loss,
        "loss/entropy_loss": entropy,
        "loss/total": value_loss + action_loss - entropy,
        #"curriculum/guided_learning": ppo_wrapper.guided_learning_prob,
        "strategies": tbl,
        #"curriculum/lr": ppo_wrapper.get_learning_rate(),
        #"curriculum/entropy_coef": ppo_wrapper.ppo_agent.entropy_coef,
    }

    logs.update(general_logs)
    return logs


def delete_run(run_to_remove: str):
    """delete_run method.

    Parameters
    ----------
    run_to_remove : str
        "<entity>/<project>/<run_id>"

    Returns
    -------
    None.

    """
    api = wandb.Api()
    run = api.run(run_to_remove)
    run.delete()
