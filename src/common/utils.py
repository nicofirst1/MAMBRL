import numpy as np
import torch
from numpy.linalg import norm


def print_current_strategy(strategies):
    reward_step_strategy, \
    reward_collision_strategy, \
    landmark_reset_strategy, \
    landmark_collision_strategy = strategies

    to_print = f"""
    reward_step_strategy: {reward_step_strategy}
    reward_collision_strategy: {reward_collision_strategy}
    landmark_reset_strategy: {landmark_reset_strategy}
    landmark_collision_strategy: {landmark_collision_strategy}

    """

    print(to_print)


def min_max_norm(val, min_, max_):
    return (val - min_) / (max_ - min_)


def rgb2gray(image: np.ndarray):
    rgb_weights = [0.2989, 0.5870, 0.1140]

    grayscale_image = np.dot(image[..., :3], rgb_weights)
    return grayscale_image


def one_hot_encode(action, n, dtype=torch.uint8):
    if not isinstance(action, torch.Tensor):
        action = torch.tensor(action)

    res = action.long().view((-1, 1))
    res = (
        torch.zeros((len(res), n))
            .to(res.device)
            .scatter(1, res, 1)
            .type(dtype)
            .to(res.device)
    )
    res = res.view((*action.shape, n))

    return res


def mas_dict2tensor(agent_dict, type=None) -> torch.Tensor:
    """
    sort agent dict and convert to tensor of type

    Params
    ------
        agent_dict:
        type:
    """

    tensor = sorted(agent_dict.items())
    if type is not None:
        tensor = [type(elem[1]) for elem in tensor]
    else:
        tensor = [elem[1] for elem in tensor]

    return torch.as_tensor(tensor)


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def is_collision(entity1, entity2):
    delta_pos = entity1.state.p_pos - entity2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))
    dist_min = entity1.size + entity2.size
    return True if dist < dist_min else False


def is_collision_border(border, agent):
    dist=norm(np.cross(border.end - border.start, border.start - agent.state.p_pos)) / norm(border.end - border.start)
    dist_min = border.size + agent.size
    return True if dist < dist_min else False


def get_distance(entity1, entity2):
    delta_pos = entity1.state.p_pos - entity2.state.p_pos
    dist = np.sqrt(np.sum(np.square(delta_pos)))

    return dist
