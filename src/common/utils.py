import numpy as np


def print_current_curriculum(curriculum):
    reward, landmark = curriculum

    r_val, r_desc = reward
    l_val, l_desc = landmark

    to_print = f"""
    Rewards:
    \t {r_val} : {r_desc}
    Landmark:
    \t {l_val} : {l_desc}
    """

    print(to_print)


def min_max_norm(val, min_, max_):
    return (val - min_) / (max_ - min_)


def rgb2gray(image: np.ndarray):
    rgb_weights = [0.2989, 0.5870, 0.1140]

    grayscale_image = np.dot(image[..., :3], rgb_weights)
    return grayscale_image
