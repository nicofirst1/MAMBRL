import torch

from src.common import mas_dict2tensor
## fixme: questa classe è da sistemare
from src.common.utils import one_hot_encode
from src.model.RolloutEncoder import RolloutEncoder


class SimulatedEnvironment:
    def __init__(self, env, model, policy, configs):
        self.env = env
        self.model = model
        self.policy = policy

        self.device = configs.device
        self.num_actions = configs.num_actions
        self.batch_size = configs.batch_size
        self.configs = configs

        self.steps = 0
        self.frames = None
        self.actions = None

        rollout_params = configs.get_rollout_encoder_configs()
        self.encoder = RolloutEncoder(**rollout_params)
        self.encoder = self.encoder.to(self.device)

    def __getattr__(self, item):
        """
        Reroute attribute selection to env
        @param item:
        @return:
        """
        if item not in self.__dict__.keys():
            return self.env.__getattribute__(item)

        return self.__dict__[item]

    def get_actions(self, agent, obs):

        if obs.ndim < 4:
            obs = obs.unsqueeze(dim=0)

        obs = obs.to(self.device)
        action, _, _ = self.policy.act(agent, obs)

        return action

    def step(self, actions):

        stacked_frames, rewards, done, infos = self.env.step(actions)

        actions = mas_dict2tensor(actions, int)

        # build new obs
        new_obs = stacked_frames.unsqueeze(dim=0).to(self.device) / 255.0
        new_action = one_hot_encode(actions, self.num_actions)
        new_action = new_action.to(self.device)

        self.model.eval()
        pred_obs = []
        pred_rews = []
        for j in range(self.configs.rollout_len):
            # update frame and actions
            self.frames = torch.concat([self.frames, new_obs], dim=0)
            self.frames = self.frames[1:]

            self.actions = torch.concat([self.actions, new_action], dim=0)
            self.actions = self.actions[1:]

            with torch.no_grad():
                new_obs, pred_rew, pred_values = self.model(self.frames, self.actions)

            new_obs = torch.argmax(new_obs, dim=1)

            # append to pred list
            pred_obs.append(new_obs)
            pred_rews.append(pred_rew)

            # get last, normalize and add fake batch dimension for stack
            new_obs = new_obs[-1] / 255
            new_obs = new_obs.unsqueeze(dim=0)

            # get new action given pred frame with policy
            # fixme: needs multi-agent support
            new_action, _, _ = self.policy.act("agent_0", new_obs)
            new_action = one_hot_encode(new_action, self.num_actions)
            new_action = new_action.to(self.device).unsqueeze(dim=0)

        pred_obs = torch.stack(pred_obs) / 255
        pred_rews = torch.stack(pred_rews)
        features = self.encoder(pred_obs, pred_rews)

        return stacked_frames, rewards, done, features

    def reset(self):
        observation = self.env.reset()
        zero_frame = torch.zeros(observation.shape)
        self.frames = torch.stack([zero_frame, zero_frame, zero_frame, observation])
        self.actions = torch.zeros(self.batch_size, self.num_actions)

        self.frames = self.frames.to(self.device)
        self.actions = self.actions.to(self.device)

        return observation

    def get_initial_frame(self):
        return self.env.initial_frame
