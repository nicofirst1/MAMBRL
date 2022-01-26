import math
from typing import Dict

import numpy as np
import torch

from src.common import print_current_strategy


class StepScheduler:

    def __init__(self, values_list, epochs, set_fn, step_list=None):
        self.values_list = values_list
        self.epochs = epochs
        self.set_fn = set_fn

        if step_list is not None:
            assert len(values_list) == len(step_list)
        else:
            step_list = list(range(0, epochs, epochs // len(values_list)))

        self.step_list = step_list

    def update_step(self, step):

        if step in self.step_list:
            idx = self.step_list.index(step)
            value = self.values_list[idx]

            if isinstance(value, dict):
                self.set_fn(**value)
            else:
                self.set_fn(value)
            return value


class CurriculumScheduler(StepScheduler):

    def __init__(self, get_curriculum_fn, **kwargs):
        super(CurriculumScheduler, self).__init__(**kwargs)

        self.get_curriculum_fn = get_curriculum_fn

    def update_step(self, step):
        value = super(CurriculumScheduler, self).update_step(step)
        if value is not None:
            print_current_strategy(self.get_curriculum_fn())


class LearningRateScheduler:

    def __init__(self, base_scheduler: torch.optim.lr_scheduler, optimizer_dict: Dict[str, torch.optim.Optimizer],
                 scheduler_kwargs):
        self.schedulers = [base_scheduler(
            optim, **scheduler_kwargs) for optim in optimizer_dict.values()]

    def update_step(self, step):
        for sc in self.schedulers:
            sc.step()


def exponential_decay(start_val, epochs, gamma=0.99):
    gamma = 1 - gamma
    values = []

    prev = start_val

    for idx in range(0, epochs):
        val = prev * pow(math.e, - gamma * idx)
        prev = val
        values.append(
            val
        )

    return values


def linear_decay(start_val, epochs):
    return list(np.linspace(start_val, 0, epochs))
