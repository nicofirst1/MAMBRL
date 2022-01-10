import matplotlib as mpl
import numpy as np

from PettingZoo.pettingzoo.mpe._mpe_utils.core import Entity


def colorFader(
    c1, c2, mix=0
):  # fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    return mpl.colors.to_rgb(mpl.colors.to_hex((1 - mix) * c1 + mix * c2))


class TimerLandmark(Entity):
    """Timer landmark class.
    Each landmark increases its timer by the 'increase' param per step.
    This timer is proportional (timer* penalty) to the penalty agents get at each turn.
    The timer resets when an agent lands upon it and the timer starts from zero.
    So the longer a landmark stays untouched the worse the penalty gets."""

    colors = [
        colorFader(np.array([0, 1, 0]), np.array([1, 0, 0]), x // 100)
        for x in range(0, 100)
    ]

    def __init__(self, np_random):
        super().__init__()
        self.np_random = np_random

    def get_random_pos(self, world):
        eps = 0.5

        return self.np_random.uniform(
            -world.max_size, world.max_size - eps, world.dim_p
        )

    def get_random_size(self):
        eps = 0.5

        return self.np_random.uniform(1 + eps, 1 - eps, 1)

    def reset(self, world, size=None, position=None):
        self.color = np.array([0, 1.0, 0])

        if position is not None:
            self.state.p_pos = position
        else:
            self.state.p_pos = self.get_random_pos(world)

        if size is not None:
            self.size = size
        else:
            self.size = self.get_random_size()
        self.state.p_vel = np.zeros(world.dim_p)
