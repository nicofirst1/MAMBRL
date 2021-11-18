import matplotlib as mpl
import numpy as np
from pettingzoo.mpe._mpe_utils.core import Entity


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
        colorFader(np.array([0, 1, 0]), np.array([1, 0, 0]), x / 100)
        for x in range(0, 100)
    ]

    def __init__(self, increase=0.1):
        super().__init__()
        self.timer = 0
        self.increase = increase
        self.counter = 0

    def reset(self, world, np_random):
        self.timer = 0
        self.color = np.array([0, 1.0, 0])
        self.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p)
        self.state.p_vel = np.zeros(world.dim_p)

    def step(self):
        self.counter += 1
        self.timer = self.increase * self.counter

        if self.counter >= len(self.colors):
            self.color = self.colors[-1]
        else:
            self.color = self.colors[self.counter]

    def reset_timer(self):
        self.timer = 0
        self.counter = 0
