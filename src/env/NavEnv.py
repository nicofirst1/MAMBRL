import math
from typing import Dict, Tuple

import numpy as np
import torch
from kornia import get_gaussian_kernel2d
from torch import nn, conv2d
from torchvision.utils import draw_bounding_boxes

from PettingZoo.pettingzoo.mpe._mpe_utils import rendering
from PettingZoo.pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv
from .Scenario import CollectLandmarkScenario
from ..common.utils import rgb2gray, get_distance


class NavEnv(SimpleEnv):
    def __init__(
            self,
            scenario_kwargs: Dict,
            horizon,
            continuous_actions: bool,
            gray_scale=False,
            frame_shape=None,
            visible=False,
            agents_positions=None,
            landmarks_positions=None,
            observation_radius=1,
            smooth_obs_mask=False,


    ):
        """
        This class has to manage the interaction between the agents in an environment.
        The env is made of N agents and M landmarks.
        The goal of the agents is to get to as many landmarks as possible in the shortest time
        Args:
            scenario_kwargs: dict of keyward for scenario initialization
            horizon: max steps before __all_done__=true
            continuous_actions: wether to use continous or discrete action
            gray_scale: if to convert obs to gray scale
            frame_shape: shape of a single frame
        """

        self.seed()
        scenario = CollectLandmarkScenario(**scenario_kwargs, np_random=self.np_random)
        world = scenario.make_world(landmarks_positions, agents_positions)

        super().__init__(
            scenario,
            world,
            max_cycles=horizon,
            continuous_actions=continuous_actions,
            local_ratio=None,
        )  # color_entities=TimerLandmark

        self.frame_shape = frame_shape
        self.render_geoms = None
        self.gray_scale = gray_scale

        self.observation_radius = observation_radius
        self.smooth_obs_mask = smooth_obs_mask

        self.agent_selection = None
        self.agents_dict = {agent.name: agent for agent in world.agents}

        self.viewer = rendering.Viewer(frame_shape[1], frame_shape[2], visible=visible)
        self.viewer.set_max_size(scenario_kwargs["max_size"])

    def set_strategy(self, **kwargs):
        self.scenario.set_strategy(**kwargs)

    def set_landmarks_pos(self, landmarks_pos):
        self.scenario.set_landmarks_pos(world=self.world, landmarks_positions=landmarks_pos)

    def set_agents_pos(self, *agents_pos):
        self.scenario.set_agents_pos(self.world, agents_pos)

    def get_current_strategy(self) -> Tuple[str, str, str, str]:
        return self.scenario.get_current_strategy()

    def get_strategies(self):
        return self.scenario.get_descriptive_strategy()

    def reset(self):
        super(NavEnv, self).reset()
        return self.observe()

    @property
    def action_meaning_dict(self):
        return {0: "stop", 1: "left", 2: "right", 3: "up", 4: "down"}

    def observe(self, agent="") -> Dict[str, torch.Tensor]:
        """
        Get an image of the game
        Args:
            agent: All observation are the same here

        Returns: returns a dict of images, one for each agent, as a torch tensor of size [channels, width, height]

        """
        obs_dict = {}
        observation = self.render()
        if observation is not None:

            observation = torch.from_numpy(observation.copy())
            observation = observation.float()

            # move channel on second dimension if present, else add 1
            if len(observation.shape) == 3:
                observation = observation.permute(2, 0, 1)
            else:
                observation = observation.unsqueeze(dim=0)

            if self.gray_scale:
                observation = rgb2gray(observation)
                observation = np.expand_dims(observation, axis=0)

            for agent_id in self.agents:
                idx = self.agents.index(agent_id)
                agent_pos = self.world.agents[idx].state.p_pos.copy()
                obs_dict[agent_id] = mask_observation(observation, agent_pos, world_size=self.world.max_size,
                                                      radius_perc=self.observation_radius,
                                                      use_conv=self.smooth_obs_mask)

        return obs_dict["agent_0"]

    def step(self, actions: Dict[str, int]) -> Tuple[torch.Tensor, Dict[str, int], Dict[str, bool], Dict[str, Dict]]:
        """
        Takes a step in the environment.
        All the agents act simultaneously and the observation are collected
        Args:
            actions: dictionary mapping angent name to an action

        Returns: all returns are dict mapping agent string to a value
            observation : the observed window as a torch.Tensor
            rewards: a reward as an int
            dones: if the agent is done or not
            infos: additional infos on the agent, such as its position

        """

        for agent_id, action in actions.items():
            self.agent_selection = agent_id
            super(NavEnv, self).step(action)

        self.steps += 1

        # perform a landmark step
        [land.step() for land in self.world.landmarks]

        self.dones["__all__"] = False
        if self.steps >= self.max_cycles:
            self.dones["__all__"] = True

        # update landmarks status
        removed = self.scenario.remove_collided_landmarks(self.world)
        if removed:
            self.render_geoms = None

        if self.scenario.num_landmarks <= 0:
            self.dones["__all__"] = True

        if self.dones["__all__"]:
            self.dones = {k: True for k in self.dones.keys()}

        self.infos = {k: self.agents_dict[k].state for k in self.infos.keys()}

        observation = self.observe()
        return observation, self.rewards, self.dones, self.infos

    def optimal_action(self, agent):
        """
        Perform an optimal action for an agent
        :param agent:
        :return:
            rew : the reward associated with the action
            action: the optimal action
            probs : a fake logprob_action
        """

        agent_id = self.agents.index(agent)
        agent = self.world.agents[agent_id]

        landmarks = self.world.landmarks

        mind_dist = 9999
        min_idx = -1
        for idx, land in enumerate(landmarks):
            dist = get_distance(agent, land)

            if dist < mind_dist:
                mind_dist = dist
                min_idx = idx

        #############################
        # Estimate optimal action
        #############################
        closest_land = landmarks[min_idx]

        agent_pos = agent.state.p_pos
        land_pos = closest_land.state.p_pos

        relative_pos = agent_pos - land_pos
        farthest_axis = np.argmax(abs(relative_pos))

        # x axis
        if farthest_axis == 0:
            if relative_pos[farthest_axis] > 0:
                # move right
                action = 1
            else:
                # move left
                action = 2

        # y axis
        else:
            if relative_pos[farthest_axis] > 0:
                # move up
                action = 3

            else:
                # move right
                action = 4

        #############################
        # Build fake log probs
        #############################

        probs = torch.zeros((1, 5))
        probs[0, action] = 1
        probs = torch.log_softmax(probs, dim=1)

        return action, probs


def get_env(kwargs: Dict) -> NavEnv:
    """Initialize rawEnv and wrap it in parallel petting zoo."""
    env = NavEnv(**kwargs)
    return env


def gaus_filter(image_size):
    # Set these to whatever you want for your gaussian filter
    kernel_size = 31
    if kernel_size % 2 == 0: kernel_size += 1
    padding = kernel_size // 2
    stride = 1
    sigma = 1
    channels = 3

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False, padding=padding,
                                stride=stride)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


def norm_filter(length, norm_ord=2, norm_func=lambda n: np.exp(-n), clip=True):
    arr = np.indices((length, length)) - ((length - 1) / 2)
    func1d = lambda x: norm_func(np.linalg.norm(x, ord=norm_ord))
    result = np.apply_along_axis(func1d, axis=0, arr=arr)
    if clip:
        bound = np.amax(np.amin(result, axis=0), axis=0)
        result *= np.logical_or(result >= bound, np.isclose(result, bound, atol=0))
    return result


def mask_observation(observation: torch.Tensor, agent_pos, world_size: int, radius_perc=0.4, use_conv=False,
                     use_cuda=True):
    """
    Apply limited field of view for world observation and agent position
    @param observation: an image of [channels, w, h]
    @param agent_pos: a vector of size 2, describing the agent position [y,x]
    @param world_size: the max world size
    @param radius_perc: percentage of field of view, 0 is none, 1 is whole world
    @param use_conv: if to use gaussian smoothing on image, for a fog of war effect
    @param use_cuda: should be true if use_conv is true, speed up conv operation.
    @return: masked observation of size [c,w,h]
    """

    assert 0 <= radius_perc or radius_perc <=1, f"Radius observation perc should be in range [0,1], got {radius_perc}"

    if radius_perc == 1:
        # if radius is 1, then no need to apply mask
        return observation

    # ratio is used to estimate agent position in image
    ratio = observation.shape[1] / (2 * world_size)

    # invert y axis and translate to have min (0,0)
    agent_pos[1] *= -1
    agent_pos += world_size

    # estimate radius and mask
    radius = world_size * radius_perc
    bbox = [
        agent_pos[0] - radius, agent_pos[1] - radius,
        agent_pos[0] + radius, agent_pos[1] + radius
    ]

    bbox = torch.as_tensor(bbox)
    bbox *= ratio
    bbox = bbox.unsqueeze(dim=0).long()

    # use torch draw bounding box to drow rectangle around agent view
    mask = draw_bounding_boxes(
        torch.zeros(observation.shape).to(torch.uint8),
        bbox,
        colors=(255, 255, 255),
        fill=True,
        width=0)

    if use_conv:

        # use a convolutional pass with a gaussian filter to simulate fog around field of view
        mask = mask.unsqueeze(dim=0).float()

        # the kernel size must be at least half of the image, this becomes computationally intensive for big images
        kernel_size = observation.shape[1] // 2
        sigma = 1e2

        # kernel must be odd
        if kernel_size % 2 == 0: kernel_size += 1

        filter = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))
        # Reshape to 2d depthwise convolutional weight
        filter = filter.view(1, 1, kernel_size, kernel_size)
        filter = filter.repeat(3, 3, 1, 1)

        if use_cuda:
            # speed up computation with cuda
            filter = filter.to("cuda")
            mask = mask.to("cuda")

        mask = conv2d(mask, filter, padding="same")
        mask = mask.squeeze().cpu() / 255.


    else:
        # use bool mask
        mask = mask.to(torch.bool)
    res = mask * observation

    # # visualize
    # from PIL import Image
    # p=res.permute(1,2,0).to(torch.uint8).numpy()
    # Image.fromarray(p).show()

    return res
