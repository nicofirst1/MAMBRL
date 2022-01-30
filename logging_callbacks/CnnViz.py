import random
from collections import OrderedDict
from functools import reduce

import numpy
import numpy as np
import scipy.ndimage as nd
import torch
import torchvision
from PIL import Image, ImageDraw
from torchvision.models import ResNet
from torchvision.transforms import transforms

from lucent.misc import ChannelReducer
from lucent.modelzoo import get_model_layers
from lucent.optvis import param, objectives, render, transform
from lucent.optvis.render import get_layer_activ
from torch import nn

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


@objectives.wrap_objective()
def dot_compare(layer, acts, batch=1):
    def inner(T):
        pred = T(layer)[batch]
        return -(pred * acts).sum(dim=0, keepdims=True).mean()

    return inner


@objectives.wrap_objective()
def dot_compare_fi(layer, batch=1, cossim_pow=0):
    def inner(T):
        dot = (T(layer)[batch] * T(layer)[0]).sum()
        mag = torch.sqrt(torch.sum(T(layer)[0] ** 2))
        cossim = dot / (1e-6 + mag)
        return -dot * cossim ** cossim_pow

    return inner


def feature_inversion(img, model, device, layer=None, n_steps=64, cossim_pow=0.0):
    # Convert image to torch.tensor and scale image
    img = torch.tensor(img).to(device)
    upsample = torch.nn.Upsample(img.shape[-1])
    img = upsample(img)

    obj = objectives.Objective.sum([
        1.0 * dot_compare_fi(layer, cossim_pow=cossim_pow),
        objectives.blur_input_each_step(),
    ])

    # Initialize parameterized input and stack with target image
    # to be accessed in the objective function
    params, image_f = param.image(img.shape[-1])

    def stacked_param_f():
        return params, lambda: torch.stack([image_f()[0], img])

    transforms = [
        transform.pad(8, mode='constant', constant_value=.5),
        transform.jitter(8),
        transform.random_scale([0.9, 0.95, 1.05, 1.1] + [1] * 4),
        transform.random_rotate(list(range(-5, 5)) + [0] * 5),
        transform.jitter(2),
    ]

    res = render.render_vis(model, obj, stacked_param_f, transforms=transforms, thresholds=(n_steps,), show_image=False,
                            progress=False)

    return res


class CnnViz:

    def __init__(self, fe, linear, name, device):

        self.feature_extractor = fe
        self.linear = linear
        self.device = device
        self.name = name

        if isinstance(fe, ResNet):

            self.preprocess = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        else:
            self.preprocess=nn.Identity()

        conv_layers = get_model_layers(fe)
        conv_layers = [x for x in conv_layers if "conv" in x and "activ" not in x]
        self.conv_layers = conv_layers

    def visualize(self, states):
        # needs device on cuda
        self.feature_extractor.to("cuda").eval()
        self.linear.to("cuda").eval()

        images = {}

        idx = random.randint(0, len(states))
        img = states[idx]/255.
        img=torch.as_tensor(img)

        img=self.preprocess(img)
        img=numpy.asarray((img))

        #images.update(self.most_active_patch(img))
        images.update(self.linear_optim(img))
        #print("Linear optim viz done")
        images.update(self.render_activation_grid(img))
        # print("render_activation_gridviz done")
        #images.update(self.aligned_interpolation())
        #print("aligned_interpolation viz done")
        #images.update(self.combined_neurons())
        # print("combined_neurons viz done")
        #images.update(self.negative_channel_viz())
        #print("negative_channel_viz viz done")

        self.feature_extractor.to(self.device)
        self.linear.to(self.device)

        images = {f"{self.name}/{k}": v for k, v in images.items()}

        return images

    def complete_model(self):
        cm = list(self.feature_extractor._modules.items())
        cm += list(self.linear._modules.items())
        cm = OrderedDict(cm)
        cm = nn.Sequential(cm)
        return cm

    def linear_optim(self, img):

        complete_model = nn.Sequential(*[self.feature_extractor, self.linear])
        param_f = lambda: param.image(img.shape[-1], batch=3)

        res = render.render_vis(complete_model, "labels:0", param_f, show_inline=False, progress=False,
                                show_image=False,
                                fixed_image_size=img.shape[-1])
        res = make_grid(res[0])

        return {f"linear": res}

    def pos_neg(self, img):

        complete_model = self.complete_model()
        img = torch.as_tensor(img).unsqueeze(dim=0).to("cuda").float()
        lay= self.conv_layers[-1]

        def objective_func(model):
            # shape: (batch_size, layer_channels, cell_layer_height, cell_layer_width)
            pred = model(lay)
            # use the sampled indices from `sample` to get the corresponding targets
            target = acts[sample.inds].to(pred.device)
            # shape: (batch_size, layer_channels, 1, 1)
            target = target.view(target.shape[0], target.shape[1], 1, 1)
            dot = (pred * target).sum(dim=1).mean()
            return -dot

        results = render.render_vis(
            complete_model,
            obj,
            param_f,
            thresholds=(n_steps,),
            show_image=False,
            progress=False,
            fixed_image_size=cell_image_size,
        )

        grads = grads.cpu().numpy()
        acts = acts.cpu().numpy().squeeze()
        reducer = ChannelReducer(n_components=3, reduction_alg="PCA")

        reducer.fit(grads)
        channel_dirs = reducer._reducer.components_

        attr_reduced = reducer.transform(np.maximum(grads, 0)) - reducer.transform(
            np.maximum(-grads, 0))  # transform the positive and negative parts separately
        nmf_norms = channel_dirs.sum(-1)
        attr_reduced *= nmf_norms[
            None, None]  # multiply by the norms of the NMF directions, since the magnitudes of the NMF directions are not relevant
        attr_reduced /= np.median(attr_reduced.max(axis=(-3, -2,
                                                         -1)))  # globally normalize by the median max value to make the visualization balanced (a bit of a hack)
        attr_pos = np.maximum(attr_reduced, 0)
        attr_pos = conv2d(attr_pos, norm_filter(attr_pos.shape[-1]))
        nmf.acts_reduced = attr_pos
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

    def feature_inversion(self, img):

        layers = self.conv_layers
        images = {}
        for layer in layers:
            print(layer)
            res = feature_inversion(img, model=self.feature_extractor, device="cuda", layer=layer, cossim_pow=1)
            images[f"fi-{layer}"] = make_grid(res[0])

        return images

    def render_activation_grid(self, img):

        model = self.feature_extractor
        device = 'cuda'
        layers = self.conv_layers[-1]
        cell_image_size = 30
        n_groups = 6
        n_steps = 16
        batch_size = 64
        images = {}

        if not isinstance(layers, list):
            layers=[layers]

        for lay in layers:
            # First wee need, to normalize and resize the image
            img = torch.tensor(img).to(device)

            transforms = transform.standard_transforms.copy() + [
                transform.normalize(),
                torch.nn.Upsample(size=img.shape[-1], mode="bilinear", align_corners=True),
            ]
            transforms_f = transform.compose(transforms)
            # shape: (1, 3, original height of img, original width of img)
            if img.ndim < 4:
                img = img.unsqueeze(0).float()
            # shape: (1, 3, 224, 224)
            img = transforms_f(img)

            # Here we compute the activations of the layer `layer` using `img` as input
            # shape: (layer_channels, layer_height, layer_width), the shape depends on the layer
            acts = get_layer_activ(model, lay, img)[0]
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
            images[f"act_grid-{lay}"] = grid

        return images

    def aligned_interpolation(self):
        def full_interpolate_obj(layer1, channel1, layer2, channel2):
            middle_lay=len(self.conv_layers)//2
            interpolation_objective = objectives.channel_interpolate(layer1, channel1, layer2, channel2)
            alignment_objective = objectives.alignment(self.conv_layers[middle_lay], decay_ratio=5)  # encourage similarity in this layer
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

        obj = full_interpolate_obj(self.conv_layers[0], 0, self.conv_layers[-1], -1)
        sequence = render.render_vis(self.feature_extractor, obj, interpolate_param_f, transforms=transforms,
                                     show_image=False, thresholds=(32,),
                                     progress=False)

        sequence = np.transpose(sequence[0], (0, 3, 1, 2)) * 255
        return {"vid-interp": sequence}

    def negative_channel_viz(self):
        """
        Just like in Lucid, we can generate visualizations that maximize activations in both the positive and negative directions.
        """
        param_f = lambda: param.image(224, batch=2)
        images = {}
        c = -1
        for lay in self.conv_layers:

            obj = objectives.channel(lay, c, batch=1) - objectives.channel(lay, c, batch=0)
            img = render.render_vis(self.feature_extractor, obj, param_f, show_image=False, thresholds=(32,),
                                    progress=False)
            lay=lay.replace("/","_")
            images[f"ncv-{lay}_c{c}"] = make_grid(img[0])
        return images

    def combined_neurons(self):
        param_f = lambda: param.image(128, batch=3)

        # First image optimizes neuron1
        # Second image optimizes neuron2
        # Third image optimizes sum of both

        neuron2 = (self.conv_layers[-2], 0)
        neuron1 = (self.conv_layers[-1], 0)

        C = lambda neuron1, neuron2: objectives.channel(*neuron1, batch=0) + \
                                     objectives.channel(*neuron2, batch=1) + \
                                     objectives.channel(*neuron1, batch=2) + \
                                     objectives.channel(*neuron2, batch=2)

        img = render.render_vis(self.feature_extractor, C(neuron1, neuron2), param_f, show_image=False,
                                thresholds=(32,),
                                progress=False)

        images = {
            f"comb-{neuron1[0]}c{neuron1[1]}V{neuron2[0]}{neuron2[1]}": make_grid(img[0])
        }

        return images

    def most_active_patch(self, img):
        """
        Get the most active patch in the states
        """
        layers = self.conv_layers
        model = self.feature_extractor

        img = torch.tensor(img).to("cuda")
        images = {}

        if img.ndim < 4:
            img = img.unsqueeze(0).float()

        for lay in layers:
            # discard first image which is incomplete
            acts, ret = get_layer_activ(model, lay, img, get_ret=True)
            acts=acts[0]

            # get most activate batch
            # get max x,y for image
            max_mag = acts.max(-1)[0]
            max_mag = max_mag.cpu()
            max_x = np.argmax(max_mag.max(-1)[0])
            max_y = np.argmax(max_mag[max_x])

            # convert states to PIL
            res_img = img.cpu().squeeze()
            res_img = np.array(res_img).astype(np.uint8)
            res_img = np.transpose(res_img, (1, 2, 0))
            res_img = Image.fromarray(res_img)

            # draw rectangle centered in max x,y
            sprite_dim = int(res_img.size[0] * 0.1)
            draw = ImageDraw.Draw(res_img, 'RGBA')

            box_w0 = max(0, max_x - sprite_dim)
            box_w1 = min(max_x + sprite_dim, res_img.size[0])

            box_h0 = max(0, max_y - sprite_dim)
            box_h1 = min(max_y + sprite_dim, res_img.size[1])

            draw.rectangle(((box_w0, box_h0), (box_w1, box_h1)), fill=(255, 255, 0, 100))
            res_img = np.asarray(res_img)
            images[f"act_patch-{lay}"] = res_img

            img=ret

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
