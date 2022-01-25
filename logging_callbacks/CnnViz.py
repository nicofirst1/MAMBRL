from collections import OrderedDict
from functools import reduce

import numpy as np
import scipy.ndimage as nd
import torch
import torchvision
from PIL import Image, ImageDraw
from lucent.misc.channel_reducer import ChannelReducer
from lucent.modelzoo.util import get_model_layers
from lucent.optvis import param, objectives, render, transform
from tqdm import tqdm

from logging_callbacks.prov import LayerNMF


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
    image = (image).astype(np.uint8)
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
                if "conv" in name or "out" in name:
                    features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)
    return features


def apply_alpha(image):
    assert image.shape[-1] == 4
    alpha = np.repeat(np.expand_dims(image[..., -1], axis=-1), 3, axis=2)
    image = image[..., :3] * alpha
    return image


def view(image):
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = apply_alpha(image)
    return Image.fromarray(image)


def make_grid(img):
    img = np.transpose(img, (0, 3, 1, 2))
    img = torchvision.utils.make_grid(torch.as_tensor(img))
    return img


@torch.no_grad()
def get_layer(model, layer, X):
    hook = render.ModuleHook(getattr(model, layer))
    model(X)
    hook.close()
    return hook.features


@objectives.wrap_objective()
def dot_compare(layer, acts, batch=1):
    def inner(T):
        pred = T(layer)[batch]
        return -(pred * acts).sum(dim=0, keepdims=True).mean()

    return inner


class CnnViz:

    def __init__(self, fe, linear, name, device):

        self.feature_extractor = fe
        # self.fe_acts = hook_model(fe)
        self.linear = linear
        # self.lin_acts = hook_model(linear)
        self.device = device
        self.name = name

    def visualize(self, states):
        # needs device on cuda
        self.feature_extractor.to("cuda").eval()

        images = {}

        images.update(self.render_activation_grid(states))
        images.update(self.aligned_interpolation())
        images.update(self.combined_neurons())
        images.update(self.negative_channel_viz())

        self.feature_extractor.to(self.device)

        return images

    def render_activation_grid(self, states):

        model = self.feature_extractor
        device = 'cuda'
        layers = ["conv_0", "conv_1", "conv_2"]
        cell_image_size = 60
        n_groups = 6
        n_steps = 64
        batch_size = 64
        img = states[2]
        images={}

        for lay in layers:
            # First wee need, to normalize and resize the image
            img = torch.tensor(img).to(device)
            normalize = (
                transform.preprocess_inceptionv1()
                if model._get_name() == "InceptionV1"
                else transform.normalize()
            )
            transforms = transform.standard_transforms.copy() + [
                normalize,
                torch.nn.Upsample(size=img.shape[-1], mode="bilinear", align_corners=True),
            ]
            transforms_f = transform.compose(transforms)
            # shape: (1, 3, original height of img, original width of img)
            if img.ndim<4:
                img = img.unsqueeze(0).float()
            # shape: (1, 3, 224, 224)
            img = transforms_f(img)

            # Here we compute the activations of the layer `layer` using `img` as input
            # shape: (layer_channels, layer_height, layer_width), the shape depends on the layer
            acts = get_layer(model, lay, img)[0]
            # shape: (layer_height, layer_width, layer_channels)
            acts = acts.permute(1, 2, 0)
            # shape: (layer_height*layer_width, layer_channels)
            acts = acts.view(-1, acts.shape[-1])
            acts_np = acts.cpu().numpy()
            nb_cells = acts.shape[0]

            # negative matrix factorization `NMF` is used to reduce the number
            # of channels to n_groups. This will be used as the following.
            # Each cell image in the grid is decomposed into a sum of
            # (n_groups+1) images. First, each cell has its own set of parameters
            #  this is what is called `cells_params` (see below). At the same time, we have
            # a of group of images of size 'n_groups', which also have their own image parametrized
            # by `groups_params`. The resulting image for a given cell in the grid
            # is the sum of its own image (parametrized by `cells_params`)
            # plus a weighted sum of the images of the group. Each each image from the group
            # is weighted by `groups[cell_index, group_idx]`. Basically, this is a way of having
            # the possibility to make cells with similar activations have a similar image, because
            # cells with similar activations will have a similar weighting for the elements
            # of the group.
            if n_groups > 0:
                reducer = ChannelReducer(n_groups, "PCA")
                groups = reducer.fit_transform(acts_np)
                groups /= groups.max(0)
            else:
                groups = np.zeros([])
            # shape: (layer_height*layer_width, n_groups)
            groups = torch.from_numpy(groups)

            # Parametrization of the images of the groups (we have 'n_groups' groups)
            groups_params, groups_image_f = param.fft_image(
                [n_groups, 3, cell_image_size, cell_image_size]
            )
            # Parametrization of the images of each cell in the grid (we have 'layer_height*layer_width' cells)
            cells_params, cells_image_f = param.fft_image(
                [nb_cells, 3, cell_image_size, cell_image_size]
            )

            # First, we need to construct the images of the grid
            # from the parameterizations

            def image_f():
                groups_images = groups_image_f()
                cells_images = cells_image_f()
                X = []
                for i in range(nb_cells):
                    x = 0.7 * cells_images[i] + 0.5 * sum(
                        groups[i, j] * groups_images[j] for j in range(n_groups)
                    )
                    X.append(x)
                X = torch.stack(X)
                return X

            # make sure the images are between 0 and 1
            image_f = param.to_valid_rgb(image_f, decorrelate=True)

            # After constructing the cells images, we sample randomly a mini-batch of cells
            # from the grid. This is to prevent memory overflow, especially if the grid
            # is large.
            def sample(image_f, batch_size):
                def f():
                    X = image_f()
                    inds = torch.randint(0, len(X), size=(batch_size,))
                    inputs = X[inds]
                    # HACK to store indices of the mini-batch, because we need them
                    # in objective func. Might be better ways to do that
                    sample.inds = inds
                    return inputs

                return f

            image_f_sampled = sample(image_f, batch_size=batch_size)

            # Now, we define the objective function

            def objective_func(model):
                # shape: (batch_size, layer_channels, cell_layer_height, cell_layer_width)
                pred = model(lay)
                # use the sampled indices from `sample` to get the corresponding targets
                target = acts[sample.inds].to(pred.device)
                # shape: (batch_size, layer_channels, 1, 1)
                target = target.view(target.shape[0], target.shape[1], 1, 1)
                dot = (pred * target).sum(dim=1).mean()
                return -dot

            obj = objectives.Objective(objective_func)

            def param_f():
                # We optimize the parametrizations of both the groups and the cells
                params = list(groups_params) + list(cells_params)
                return params, image_f_sampled

            results = render.render_vis(
                model,
                obj,
                param_f,
                thresholds=(n_steps,),
                show_image=False,
                progress=False,
                fixed_image_size=cell_image_size,
            )
            # shape: (layer_height*layer_width, 3, grid_image_size, grid_image_size)
            imgs = image_f()
            imgs = imgs.cpu().data
            imgs = imgs[:, :, 2:-2, 2:-2]
            # turn imgs into a a grid
            grid = torchvision.utils.make_grid(imgs, nrow=int(np.sqrt(nb_cells)), padding=0)
            grid = grid.permute(1, 2, 0)
            grid = grid.numpy() * 255
            images[f"act_grid/{lay}"]=grid

        return images

    def aligned_interpolation(self):
        def full_interpolate_obj(layer1, channel1, layer2, channel2):
            interpolation_objective = objectives.channel_interpolate(layer1, channel1, layer2, channel2)
            alignment_objective = objectives.alignment('conv_1', decay_ratio=5)  # encourage similarity in this layer
            return interpolation_objective + 1e-1 * alignment_objective

        def interpolate_param_f():
            # We start with a batch of FFT parameterized images
            params, image_f = param.fft_image((6, 3, 128, 128))
            # We then add a bunch of low-res interpolated tensors
            shared = [
                param.lowres_tensor((6, 3, 128, 128), (1, 3, 128 // 2, 128 // 2)),
                param.lowres_tensor((6, 3, 128, 128), (1, 3, 128 // 4, 128 // 4)),
                param.lowres_tensor((6, 3, 128, 128), (1, 3, 128 // 8, 128 // 8)),
                param.lowres_tensor((6, 3, 128, 128), (2, 3, 128 // 8, 128 // 8)),
                param.lowres_tensor((6, 3, 128, 128), (1, 3, 128 // 16, 128 // 16)),
                param.lowres_tensor((6, 3, 128, 128), (2, 3, 128 // 16, 128 // 16)),
            ]
            # Accumulate the params and outputs
            images = []
            for (p, i) in shared:
                params += p
                images.append(i)
            return params, param.to_valid_rgb(lambda: image_f() + sum([i() for i in images]), decorrelate=True)

        # Set up transforms for a nicer image
        transforms = [
            transform.pad(12, mode="constant", constant_value=.5),
            transform.jitter(8),
            transform.random_scale([.75]),
        ]

        obj = full_interpolate_obj("conv_0", 0, "conv_2", -1)
        sequence = render.render_vis(self.feature_extractor, obj, interpolate_param_f, transforms=transforms,
                                     show_image=False, thresholds=(32,),
                                     progress=False)

        sequence = np.transpose(sequence[0], (0, 3, 1, 2)) * 255
        return {"vid/interp": sequence}

    def negative_channel_viz(self):
        param_f = lambda: param.image(224, batch=2)
        images = {}
        for lay in get_model_layers(self.feature_extractor):

            if "conv" not in lay or "activ" in lay:
                continue
            channels = getattr(self.feature_extractor, lay).out_channels
            print(f"Layer {lay} with {channels} channels\n")

            for c in tqdm(range(channels)):
                obj = objectives.channel(lay, c, batch=1) - objectives.channel(lay, c, batch=0)
                img = render.render_vis(self.feature_extractor, obj, param_f, show_image=False, thresholds=(32,),
                                        progress=False)

                images[f"ncv/{lay}/{c}"] = make_grid(img[0])
        return images

    def combined_neurons(self):
        param_f = lambda: param.image(128, batch=3)

        # First image optimizes neuron1
        # Second image optimizes neuron2
        # Third image optimizes sum of both

        neuron2 = ('conv_0', 0)
        neuron1 = ('conv_1', 0)

        C = lambda neuron1, neuron2: objectives.channel(*neuron1, batch=0) + \
                                     objectives.channel(*neuron2, batch=1) + \
                                     objectives.channel(*neuron1, batch=2) + \
                                     objectives.channel(*neuron2, batch=2)

        img = render.render_vis(self.feature_extractor, C(neuron1, neuron2), param_f, show_image=False,
                                thresholds=(32,),
                                progress=False)

        images = {
            f"cams/comb/{neuron1[0]}c{neuron1[1]}V{neuron2[0]}{neuron2[1]}": make_grid(img[0])
        }

        return images

    def visualize2(self):

        images = {}

        reduction_alg = "PCA"

        # get original image input, denormalize and transpose
        states = self.fe_acts['conv_0'].input * 255
        states = states.cpu().detach().numpy()
        states = np.transpose(states, [0, 2, 3, 1])

        for name, hook in self.fe_acts.items():

            if "activ" in name:
                continue

            # get activations and grads
            acts = hook.acts
            grads = hook.grads

            if reduction_alg == "NMF":
                # if reduction is NMF zero out negative activations
                acts[acts < 0] = 0

            neuron_groups(states, acts, name, self.feature_extractor, n_groups=6, attr_classes=[])

            images[f"map/{name}"] = most_active_patch(acts, states)

            nmf = LayerNMF(acts, states, features=3,
                           reduction_alg="PCA", )  # grads=grads)  # , attr_layer_name=value_function_name)

            image = nmf.vis_dataset_thumbnail(0, num_mult=4, expand_mult=4)[0]
            image = apply_alpha(image)
            image = zoom_to(image, 200)
            images[f"thumb/{name}"] = image

            # image = nmf.vis_dataset(1, subdiv_mult=1, expand_mult=4)[0]
            # image = apply_alpha(image)
            # image = zoom_to(image, 800)
            # images[f"spatio/{name}"] = image

            # attr_reduced = nmf.transform(np.maximum(grads, 0)) - nmf.transform(
            #     np.maximum(-grads, 0))  # transform the positive and negative parts separately
            # nmf_norms = nmf.channel_dirs.sum(-1)
            # attr_reduced *= nmf_norms[
            #     None, None, None]  # multiply by the norms of the NMF directions, since the magnitudes of the NMF directions are not relevant
            # attr_reduced /= np.median(attr_reduced.max(axis=(-3, -2,
            #                                                  -1)))  # globally normalize by the median max value to make the visualization balanced (a bit of a hack)
            # attr_pos = np.maximum(attr_reduced, 0)
            # attr_pos = conv2d(attr_pos, norm_filter(attr_pos.shape[-1]))
            # nmf.acts_reduced = attr_pos
            # image = nmf.vis_dataset(1, subdiv_mult=1, expand_mult=4)[0]
            # image = zoom_to(image, 200)
            # image = apply_alpha(image)
            # images[f"grad/{name}_pos"] = image
            #
            # attr_neg = np.maximum(-attr_reduced, 0)
            # attr_neg = conv2d(attr_neg, norm_filter(attr_pos.shape[-1]))
            # nmf.acts_reduced = attr_neg
            # image = nmf.vis_dataset(1, subdiv_mult=1, expand_mult=4)[0]
            # image = zoom_to(image, 200)
            # image = apply_alpha(image)
            #
            # images[f"grad/{name}_neg"] = image
        return images


def neuron_groups(img, acts, layer, model, n_groups=6, attr_classes=[]):
    # Compute activations

    # We'll use ChannelReducer (a wrapper around scikit learn's factorization tools)
    # to apply Non-Negative Matrix factorization (NMF).
    # Compute activations

    # We'll use ChannelReducer (a wrapper around scikit learn's factorization tools)
    # to apply Non-Negative Matrix factorization (NMF).

    nmf = ChannelReducer(n_groups, "PCA")
    spatial_factors = nmf.fit_transform(acts)[0].transpose(2, 0, 1).astype("float32")
    channel_factors = nmf._reducer.components_.astype("float32")

    # Let's organize the channels based on their horizontal position in the image

    x_peak = np.argmax(spatial_factors.max(1), 1)
    ns_sorted = np.argsort(x_peak)
    spatial_factors = spatial_factors[ns_sorted]
    channel_factors = channel_factors[ns_sorted]

    # And create a feature visualziation of each group

    param_f = lambda: param.image(80, batch=n_groups)
    obj = sum(objectives.direction(layer, channel_factors[i], batch=i)
              for i in range(n_groups))
    group_icons = render.render_vis(model, obj, param_f, verbose=False)[-1]

    # We'd also like to know about attribution

    # First, let's turn each group into a vector over activations
    group_vecs = [spatial_factors[i, ..., None] * channel_factors[i]
                  for i in range(n_groups)]

    attrs = np.asarray([raw_class_group_attr(img, layer, attr_class, group_vecs)
                        for attr_class in attr_classes])


def most_active_patch(acts, states):
    """
    Get the most active patch in the states
    """

    # discard first image which is incomplete
    states = states[1:]
    acts = acts[1:]

    # get most activate batch
    acts = acts.max(0)
    max_index = acts[1][0, 0, 0]
    acts = acts[0]
    states = states[max_index]

    # get max x,y for image
    max_mag = acts.max(-1)[0]
    max_x = np.argmax(max_mag.max(-1)[0])
    max_y = np.argmax(max_mag[max_x])

    # convert states to PIL
    states = np.array(states).astype(np.uint8)
    states = Image.fromarray(states)

    # draw rectangle centered in max x,y
    sprite_dim = int(states.size[0] * 0.1)
    draw = ImageDraw.Draw(states, 'RGBA')

    box_w0 = max(0, max_x - sprite_dim)
    box_w1 = min(max_x + sprite_dim, states.size[0])

    box_h0 = max(0, max_y - sprite_dim)
    box_h1 = min(max_y + sprite_dim, states.size[1])

    draw.rectangle(((box_w0, box_h0), (box_w1, box_h1)), fill=(255, 255, 0, 100))

    return np.asarray(states)
