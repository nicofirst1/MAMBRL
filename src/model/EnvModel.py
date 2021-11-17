import torch
import torch.nn as nn

"""
    Here something is not working.. the outputs of self.conv1 and self.conv2 
    do not have the same dimension, in particular one is (1, 16, x, y) and the
    other is (1, 32, z, z). Thus, it's not possible in the forward function to
    apply the cat function. I've replaced the self.conv1 with another version
    to make it work for now, but this problem must be investigated
    
    Nico: IMO it is not important if we copy exactly the conv structure as long as it serve its purpose
     (extract features from image)
"""


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
            nn.Conv2d(n1 + n2, n3, kernel_size=1),
            nn.ReLU()
        )

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
    def __init__(self, in_shape, num_pixels, num_rewards):
        super(EnvModel, self).__init__()

        width = in_shape[1]
        height = in_shape[2]

        # fixme: imo this are way to many conv for a 32x32 image,
        #  we have 3 in each basicBlock + 1 conv + 1 if image or 2 if reward = 8/9
        self.conv = nn.Sequential(
            nn.Conv2d(8, 64, kernel_size=1), # 8 = 3 frames + 5 actions
            nn.ReLU()
        )

        self.basic_block1 = BasicBlock((64, width, height), 16, 32, 64)
        self.basic_block2 = BasicBlock((128, width, height), 16, 32, 64)

        self.image_conv = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1),
            nn.ReLU()
        )
        self.image_fc = nn.Linear(256, num_pixels)

        self.reward_conv = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU()
        )
        # fixme: qui il num rewards dipende dal gioco pacman su cui e' stato fatto il paper.
        #  In pratica hanno una lista con dentro possibili rewards per ogni azione.
        #  Noi invece (al momento) abbiamo un float... Gli approcci possono essere 2:
        #  1) rendiamo la nostra reward statica (ad ogni step puoi ricevere solo un numero finito di interi)
        #  2) Rendiamo il problema una regressione, e a quel punto rimane un solo numero
        #  (ma perche non lo hanno fatto loro?)
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
        # [batch size, features, img_w, img_h] ->[batch size, img_w, img_h, features] with permutation
        # [batch size, img_w, img_h, features] -> [whatever, 256] with view
        # fixme: why 256 is so arbitrary? Maybe bc is the pixel interval?
        image = image.permute(0, 2, 3, 1).contiguous().view(-1, 256)
        image = self.image_fc(image)

        reward = self.reward_conv(x)
        reward = reward.view(batch_size, -1)
        reward = self.reward_fc(reward)[0]

        return image, reward
