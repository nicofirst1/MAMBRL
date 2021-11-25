import torch
import torch.nn as nn


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
        self.conv3 = nn.Sequential(nn.Conv2d(n1 + n2, n3, kernel_size=1), nn.ReLU())

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


class EnvModel(nn.Module):
    def __init__(self, in_shape, num_rewards, num_frames, num_actions, num_colors):
        super(EnvModel, self).__init__()

        width = in_shape[1]
        height = in_shape[2]
        num_pixels = width * height

        num_channels = in_shape[0]

        # fixme: imo this are way to many conv for a 32x32 image,
        #  we have 3 in each basicBlock + 1 conv + 1 if image or 2 if reward = 8/9
        self.conv = nn.Sequential(
            nn.Conv2d(
                num_channels * num_frames + num_actions, 64, kernel_size=1
            ),
            nn.ReLU(),
        )

        self.basic_block1 = BasicBlock((64, width, height), 16, 32, 64)
        self.basic_block2 = BasicBlock((128, width, height), 16, 32, 64)

        self.image_conv = nn.Sequential(nn.Conv2d(192, 256, kernel_size=1), nn.ReLU())
        self.image_fc = nn.Linear(256, num_colors) #num_channels * num_pixels)

        self.reward_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
        )

        self.reward_fc = nn.Linear(64 * width * height, num_rewards)

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


