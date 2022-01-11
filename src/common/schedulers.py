from typing import Dict

import torch

from src.common import print_current_curriculum


class StepScheduler:

    def __init__(self, values_list, episodes, set_fn, step_list=None):
        self.values_list = values_list
        self.episodes = episodes
        self.set_fn = set_fn

        if step_list is not None:
            assert len(values_list) == len(step_list)
        else:
            step_list = list(range(0, episodes, episodes // len(values_list)))

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
            print_current_curriculum(self.get_curriculum_fn())


class GuidedLearningScheduler(StepScheduler):

    def update_step(self, step):
        value = super(GuidedLearningScheduler, self).update_step(step)

        if value is not None:
            print(f"\nGuided learning prob set to :{value}")


class LearningRateScheduler:

    def __init__(self, base_scheduler: torch.optim.lr_scheduler, optimizer_dict: Dict[str, torch.optim.Optimizer],
                 scheduler_kwargs):
        self.schedulers = [base_scheduler(optim, **scheduler_kwargs) for optim in optimizer_dict.values()]

    def update_step(self, step):
        for sc in self.schedulers:
            sc.step()
