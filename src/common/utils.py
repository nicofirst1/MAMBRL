import numpy as np
import torch


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
