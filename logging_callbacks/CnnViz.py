from collections import OrderedDict
from functools import reduce

import numpy as np
import scipy.ndimage as nd
import torch
from PIL import Image
from lucent.misc.channel_reducer import ChannelReducer

from logging_callbacks.prov import LayerNMF, conv2d, norm_filter


def argmax_nd(x, axes, *, max_rep=np.inf, max_rep_strict=None):
    assert max_rep > 0
    assert np.isinf(max_rep) or max_rep_strict is not None
    perm = list(range(len(x.shape)))
    for axis in reversed(axes):
        loc = perm.index(axis)
        perm = [axis] + perm[:loc] + perm[loc + 1:]
    x = x.transpose(perm)
    shape = x.shape
    axes_size = reduce(lambda a, b: a * b, shape[: len(axes)], 1)
    x = x.reshape([axes_size, -1])
    indices = np.argsort(-x, axis=0)
    result = indices[0].copy()
    counts = np.zeros(len(indices), dtype=int)
    unique_values, unique_counts = np.unique(result, return_counts=True)
    counts[unique_values] = unique_counts
    for i in range(1, len(indices) + (0 if max_rep_strict else 1)):
        order = np.argsort(x[result, range(len(result))])
        result_in_order = result[order]
        current_counts = counts.copy()
        changed = False
        for j in range(len(order)):
            value = result_in_order[j]
            if current_counts[value] > max_rep:
                pos = order[j]
                new_value = indices[i % len(indices)][pos]
                result[pos] = new_value
                current_counts[value] -= 1
                counts[value] -= 1
                counts[new_value] += 1
                changed = True
        if not changed:
            break
    result = result.reshape(shape[len(axes):])
    return np.unravel_index(result, shape[: len(axes)])


def vis_dataset_thumbnail(
        feature, *, num_mult=1, expand_mult=1, max_rep=None
):
    if max_rep is None:
        max_rep = num_mult

    acts_feature = feature
    pos_indices = argmax_nd(
        acts_feature, axes=[1, 2], max_rep=max_rep, max_rep_strict=True
    )
    acts_single = acts_feature[
        range(acts_feature.shape[0]), pos_indices[0], pos_indices[1]
    ]
    obs_indices = np.argsort(-acts_single, axis=0)[: num_mult ** 2]
    coords = np.array(list(zip(*pos_indices)), dtype=[("h", int), ("w", int)])[
        obs_indices
    ]
    indices_order = np.argsort(coords, axis=0, order=("h", "w"))
    indices_order = indices_order.reshape((num_mult, num_mult))
    for i in range(num_mult):
        indices_order[i] = indices_order[i][
            np.argsort(coords[indices_order[i]], axis=0, order="w")
        ]
    obs_indices = obs_indices[indices_order]
    poses = np.array(pos_indices).transpose()[obs_indices] + 0.5
    # self.pad_obses(expand_mult=expand_mult)
    patches = []
    patch_acts = np.zeros((num_mult, num_mult))
    patch_shapes = []
    for i in range(num_mult):
        patches.append([])
        for j in range(num_mult):
            obs_index = obs_indices[i, j]
            pos_h, pos_w = poses[i, j]
            # patch = self.get_patch(obs_index, pos_h, pos_w, expand_mult=expand_mult)
            patch = obs_index
            patches[i].append(patch)
            patch_acts[i, j] = acts_single[obs_index]
            patch_shapes.append(patch.shape)
    patch_acts_max = patch_acts.max()
    opacities = patch_acts / (1 if patch_acts_max == 0 else patch_acts_max)
    patch_min_h = np.array([s[0] for s in patch_shapes]).min()
    patch_min_w = np.array([s[1] for s in patch_shapes]).min()
    for i in range(num_mult):
        for j in range(num_mult):
            opacity = opacities[i, j][None, None, None]
            opacity = opacity.repeat(patches[i][j].shape[0], axis=0)
            opacity = opacity.repeat(patches[i][j].shape[1], axis=1)
            patches[i][j] = np.concatenate([patches[i][j], opacity], axis=-1)
            patches[i][j] = patches[i][j][:patch_min_h, :patch_min_w]
    return (
        np.concatenate(
            [np.concatenate(patches[i], axis=1) for i in range(len(patches))],
            axis=0,
        ),
        obs_indices.tolist(),
    )


def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image


def view(image):
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image ).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()


def zoom_to(img, width):
    n = width // img.shape[-2] + 1
    img = img.repeat(n, axis=-3).repeat(n, axis=-2)
    r = float(width) / img.shape[-2]
    zoom = [1] * (img.ndim - 3) + [r, r, 1]
    return nd.zoom(img, zoom, order=0, mode="nearest")


class ModuleHook:
    def __init__(self, module):

        self.fw_hook = module.register_forward_hook(self.forward_hook)
        self.bk_hook = module.register_backward_hook(self.backward_hook)
        self.module = None
        self.acts = None
        self.grads = None
        self.input = None

    def forward_hook(self, module, input, output):
        self.module = module
        self.acts = output.detach().cpu()
        self.input = input[0]

    def backward_hook(self, module, grad_input, grad_output):
        self.grads = grad_output[0].detach().cpu()


    def close(self):
        self.fw_hook.remove()
        self.bk_hook.remove()


def hook_model(model):
    features = OrderedDict()

    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if "conv" in name or "out" in name and "activ" in name:
                    features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)
    return features

def apply_alpha(image):
    assert image.shape[-1]==4
    alpha= np.repeat(np.expand_dims(image[..., -1], axis=-1), 3, axis=2)
    image = image[..., :3] * alpha
    return image

class CnnViz:

    def __init__(self, fe, linear, name, device):

        self.feature_extractor = fe
        self.fe_acts = hook_model(fe)
        self.linear = linear
        self.lin_acts = hook_model(linear)
        self.device = device
        self.name = name

    def visualize(self):
        # self.feature_extractor.train()
        # self.linear.train()
        #
        # inputs = torch.as_tensor(states) / 255.0
        # inputs = inputs.to(self.device)
        # out = self.feature_extractor(inputs)
        # out = self.linear(out)

        images = {}

        reduction_alg="PCA"

        states=self.fe_acts['conv_0'].input*255
        states=states.cpu()
        states = np.transpose(states, [0, 2, 3, 1])
        states=np.asarray(states)

        for name, hook in self.fe_acts.items():
            features = hook.acts
            grads=hook.grads

            if reduction_alg=="NMF":
                features[features < 0] = 0

            nmf = LayerNMF(features, states, features=3, reduction_alg="PCA", grads=grads)  # , attr_layer_name=value_function_name)


            image = nmf.vis_dataset_thumbnail(0, num_mult=4, expand_mult=4)[0]
            image = apply_alpha(image)
            image = zoom_to(image, 200)
            images[f"thumb/{name}"] = image

            image = nmf.vis_dataset(1, subdiv_mult=1, expand_mult=4)[0]
            image = apply_alpha(image)
            image = zoom_to(image, 800)
            images[f"spatio/{name}"] = image

            attr_reduced = nmf.transform(np.maximum(grads, 0)) - nmf.transform(
                np.maximum(-grads, 0))  # transform the positive and negative parts separately
            nmf_norms = nmf.channel_dirs.sum(-1)
            attr_reduced *= nmf_norms[
                None, None, None]  # multiply by the norms of the NMF directions, since the magnitudes of the NMF directions are not relevant
            attr_reduced /= np.median(attr_reduced.max(axis=(-3, -2,
                                                             -1)))  # globally normalize by the median max value to make the visualization balanced (a bit of a hack)
            attr_pos = np.maximum(attr_reduced, 0)
            attr_pos = conv2d(attr_pos, norm_filter(attr_pos.shape[-1]))
            nmf.acts_reduced=attr_pos
            image = nmf.vis_dataset(1, subdiv_mult=1, expand_mult=4)[0]
            image = zoom_to(image, 200)
            image = apply_alpha(image)
            images[f"grad/{name}_pos"] = image

            attr_neg = np.maximum(-attr_reduced, 0)
            attr_neg = conv2d(attr_neg, norm_filter(attr_pos.shape[-1]))
            nmf.acts_reduced = attr_neg
            image = nmf.vis_dataset(1, subdiv_mult=1, expand_mult=4)[0]
            image = zoom_to(image, 200)
            image = apply_alpha(image)

            images[f"grad/{name}_neg"] = image
        return images
