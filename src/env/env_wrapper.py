from typing import Tuple

import torch

from env.envs import CollectLandmarkEnv
from model.utils import one_hot_encode

class EnvWrapper:

    def __init__(self, env: CollectLandmarkEnv, frame_shape, num_stacked_frames, device):

        self.env = env
        self.buffer = []

        self.initial_frame = None
        self.channel_size = frame_shape[0]
        self.stacked_frames = torch.zeros(self.channel_size * num_stacked_frames, *frame_shape[1:])

        self.obs_shape = self.stacked_frames.shape
        self.action_space = self.env.action_spaces["agent_0"].n

        self.agents = self.env.agents_dict
        self.device = device

    def set_curriculum(self, reward: int = None, landmark: int = None):
        self.env.set_curriculum(reward, landmark)

    def get_curriculum(self) -> Tuple[Tuple[int, str], Tuple[int, str]]:
        return self.env.get_curriculum()

    def reset(self):
        observation = self.env.reset()

        num_frames = self.stacked_frames.shape[0] // self.channel_size
        obs = observation.repeat(num_frames, 1, 1)

        self.stacked_frames = obs
        if self.initial_frame is None:
            self.initial_frame = self.stacked_frames.clone()

        return self.stacked_frames

    def step(self, actions):
        new_obs, rewards, done, infos = self.env.step(actions)
        self.add_interaction(torch.tensor(actions["agent_0"]), torch.tensor(rewards["agent_0"]), new_obs, done["__all__"])

        self.stacked_frames = torch.cat((self.stacked_frames[self.channel_size:], new_obs), dim=0)
        if done["__all__"]:
            value = torch.tensor(0.).to(self.device)
            self.buffer[0][5] = value
            index = 0
            while True:
                ## fixme: nell'add_interaction c'è il +1 e qua -1.. da capire se serve più avanti quel +1
                # value = (self.buffer[index][2] - 1).to(self.device) + 0.998 * value
                value = (self.buffer[index][2]).to(self.device) + 0.998 * value
                self.buffer[index][5] = value
                index += 1

                if self.buffer[index][4] == 1:
                    break

        return self.stacked_frames, rewards, done, infos

    def add_interaction(self, actions, rewards, new_obs, done):
        current_obs = self.stacked_frames.squeeze().byte().to(self.device)
        action = one_hot_encode(actions, self.action_space).to(self.device)
        #reward = (rewards.squeeze() + 1).byte().to(self.device) ## fixme: perché c'è il +1?
        reward = (rewards.squeeze() + 1).byte().to(self.device)
        new_obs = new_obs.squeeze().byte().to(self.device)
        done = torch.tensor(done, dtype=torch.uint8).to(self.device)
        self.buffer.append([current_obs, action, reward, new_obs, done, None])