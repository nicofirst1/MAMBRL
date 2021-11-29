import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import rgb2gray


class BasicBlock(nn.Module):
    """
    Basic image block. Extract features from image 3 different convolutions
    """

    def __init__(self, in_shape, n1, n2, n3):
        super(BasicBlock, self).__init__()

        self.in_shape = in_shape
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3

        self.maxpool = nn.MaxPool2d(kernel_size=in_shape[1:])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n1, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n1, n1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_shape[0] * 2, n2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(n2, n2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n1 + n2, n3, kernel_size=1), nn.ReLU())

    def forward(self, inputs):
        x = self.pool_and_inject(inputs)
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], 1)
        x = self.conv3(x)
        x = torch.cat([x, inputs], 1)
        return x

    def pool_and_inject(self, x):
        pooled = self.maxpool(x)
        tiled = pooled.expand((x.size(0),) + self.in_shape)
        out = torch.cat([tiled, x], 1)
        return out


def target_to_pix(color_index, gray_scale=False):
    color_index = [torch.as_tensor(x) for x in color_index]

    def inner(imagined_states):
        #fixme: anche questo e' sbagliato
        batch_size = imagined_states.shape[0]
        image_shape = imagined_states.shape[-2:]

        new_imagined_state = torch.zeros([batch_size, *image_shape, 3]).long()

        # remove channel dim since is 1
        imagined_states = imagined_states.squeeze(1)

        for c in range(len(color_index)):
            indices = imagined_states == c
            new_imagined_state[indices] = color_index[c]

        new_imagined_state = new_imagined_state.view(
            batch_size, 3, *image_shape)

        if False:  # debug, show image
            from PIL import Image

            img = new_imagined_state[0].cpu().view(32, 32, 3)
            img = Image.fromarray(img.numpy(), mode="RGB")
            img.show()

        if gray_scale:
            new_imagined_state = rgb2gray(new_imagined_state, dimension=1)
            new_imagined_state=new_imagined_state.ceil()

        return new_imagined_state

    return inner


class EnvModel(nn.Module):
    def __init__(self, in_shape, reward_range, num_frames, num_actions, num_colors, target2pix):
        super(EnvModel, self).__init__()

        width = in_shape[1]
        height = in_shape[2]
        num_pixels = width * height

        num_channels = in_shape[0]

        self.num_actions = num_actions
        self.in_shape = in_shape
        self.target2pix = target2pix
        self.reward_range= reward_range

        # fixme: imo this are way to many conv for a 32x32 image,
        #  we have 3 in each basicBlock + 1 conv + 1 if image or 2 if reward = 8/9
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels * num_frames + num_actions, 64, kernel_size=1),
            nn.ReLU(),
        )

        self.basic_block1 = BasicBlock((64, width, height), 16, 32, 64)
        self.basic_block2 = BasicBlock((128, width, height), 16, 32, 64)

        self.image_conv = nn.Sequential(nn.Conv2d(192, 256, kernel_size=1), nn.ReLU())
        self.image_fc = nn.Linear(256, num_colors)  # num_channels * num_pixels)

        self.reward_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
        )

        self.reward_fc = nn.Linear(64 * width * height, len(reward_range))

    def forward(self, inputs):
        """
        Given the current input(a concatenation of the observed frames and the propagated actions)
        predict next frame and reward
        """
        batch_size = inputs.size(0)

        x = self.conv(inputs)
        x = self.basic_block1(x)
        x = self.basic_block2(x)
        # [batch size, features, img_w, img_h]

        image = self.image_conv(x)

        # [batch size, features, img_w, img_h] -> [batch size, img_w, img_h, features] with permutation
        # [batch size, img_w, img_h, features] -> [whatever, 256] with view
        image = image.permute(0, 2, 3, 1).contiguous().view(-1, 256)
        image = self.image_fc(image)

        reward = self.reward_conv(x)
        reward = reward.view(batch_size, -1)
        reward = self.reward_fc(reward)

        return image, reward

    def full_pipeline(self, actions: torch.Tensor, states: torch.Tensor):
        """
        Create the onhot action image and feed it to the forward method.
        Compute softmax and reshape bach to right dims
        """
        batch_size = actions.shape[0]

        onehot_action = torch.zeros(
            batch_size, self.num_actions, *self.in_shape[1:]
        )

        # fixme: this does not work
        onehot_action[:, actions] = 1
        onehot_action = onehot_action.to(states.device)
        inputs = torch.cat([states, onehot_action], 1)
        # inputs = inputs.to(self.device)

        imagined_state, reward = self.forward(inputs)

        imagined_state = F.softmax(imagined_state, dim=1).max(dim=1)[1]
        imagined_state = imagined_state.view(
            batch_size
            , *self.in_shape[1:])
        imagined_state = self.target2pix(imagined_state)
        imagined_state= imagined_state.to(states.device)

        imagined_reward = F.softmax(reward, dim=1).max(dim=1)[1]
        imagined_reward= [self.reward_range[x] for x in imagined_reward]
        imagined_reward=torch.as_tensor(imagined_reward).to(states.device)

        return imagined_state, imagined_reward
