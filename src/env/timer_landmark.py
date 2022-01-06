import numpy as np
import matplotlib as mpl

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

    def __init__(self):
        super().__init__()

    def reset(self, world, np_random):
        self.color = np.array([0, 1.0, 0])

        eps = 0.5

        self.state.p_pos = np_random.uniform(-world.max_size + eps, world.max_size - eps, world.dim_p)
        self.state.p_vel = np.zeros(world.dim_p)
