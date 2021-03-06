from typing import Dict, Tuple

import PIL.Image
import numpy as np
import torch

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

    def observe(self, agent="") -> torch.Tensor:
        """
        Get an image of the game
        Args:
            agent: All observation are the same here

        Returns: returns an image as a torch tensor of size [channels, width, height]

        """
        observation = self.render()
        if observation is not None:

            # if the observation is not of frameshape, resize.
            # This sometimes happen on specific machines, the cause is unkown
            if observation.shape[:1] != self.frame_shape[1:]:
                observation = PIL.Image.fromarray(observation)
                observation = observation.resize(self.frame_shape[1:])
                observation = np.asarray(observation)

            observation = torch.from_numpy(observation.copy())
            observation = observation.float()

            # move channel on second dimension if present, else add 1
            if len(observation.shape) == 3:
                observation = observation.permute(2, 1, 0)
            else:
                observation = observation.unsqueeze(dim=0)

            if self.gray_scale:
                observation = rgb2gray(observation)
                observation = np.expand_dims(observation, axis=0)

            # normalize observations
            #observation /= 255.0

        return observation

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
