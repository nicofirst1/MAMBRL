import numpy as np
from functools import reduce, partial
import scipy.ndimage as nd
import torch
from lucent.misc.channel_reducer import ChannelReducer



def conv2d(image, kernel, bias=0):
    assert image.ndim == 4, (
        "input_ must have 4 dimensions "
        "corresponding to batch, height, width and channels"
    )
    assert (
            kernel.ndim == 2
    ), "filter_ must have 2 dimensions and will be applied channelwise"

    kernel = np.dot(
        kernel.astype(image.dtype),
        np.eye(image.shape[-1], dtype=image.dtype),
    )

    m, n = kernel.shape
    if (m == n):
        batch,y, x,c = image.shape
        y = y - m + 1
        x = x - m + 1
        new_image = np.zeros((batch, y, x,c))
        for b in range(batch):
            for i in range(y):
                for j in range(x):
                    new_image[b][i][j] = np.sum(image[b, i:i + m, j:j + m] * kernel) + bias


    return np.resize(new_image, image.shape)

def norm_filter(length, norm_ord=2, norm_func=lambda n: np.exp(-n), clip=True):
    arr = np.indices((length, length)) - ((length - 1) / 2)
    func1d = lambda x: norm_func(np.linalg.norm(x, ord=norm_ord))
    result = np.apply_along_axis(func1d, axis=0, arr=arr)
    if clip:
        bound = np.amax(np.amin(result, axis=0), axis=0)
        result *= np.logical_or(result >= bound, np.isclose(result, bound, atol=0))
    return result


class VanillaBackprop():
    """
        Produces gradients generated with vanilla back propagation from the image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = {}
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()

    def hook_layers(self):
        def hook_function(name, module, grad_in, grad_out):

            self.gradients[name] = grad_in[0]

        # Register hook to the first layer
        for name, layer in self.model._modules.items():
            hf=partial(hook_function,name)
            layer.register_backward_hook(hf)

    def generate_gradients(self, input_image, target_class):
        # Forward
        model_output = self.model(input_image)
        # Zero grads
        self.model.zero_grad()
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        one_hot_output= one_hot_output.to(input_image.device)
        # Backward pass
        model_output.backward(gradient=one_hot_output)

        #normalize gradients
        for k, grad in self.gradients.items():
            grad -= grad.min()
            grad /= grad.max()
            self.gradients[k] = grad

        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)

        return self.gradients
