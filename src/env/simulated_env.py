import torch

from model.utils import mas_dict2tensor


## fixme: questa classe è da sistemare


class SimulatedEnvironment:
    def __init__(self, env, model, num_actions, device):
        self.env = env
        self.model = model
        self.device = device
        self.num_actions = num_actions
        self.steps = 0
        self.frames = None
        self.actions = None

    def step(self, actions):
        actions = mas_dict2tensor(actions, int)
        actions = actions.to(self.device)

        if self.steps == 0:
            self.frames = torch.stack(self.frames).view(1, 12, 96, 96)
            self.frames = self.frames.float() / 255
            self.frames = self.frames.to(self.device)

        self.steps += 1

        # actions = one_hot_encode(actions, self.num_actions, dtype=torch.float32)
        # actions = actions.to(self.device)

        self.model.eval()
        with torch.no_grad():
            new_states, rewards, values = self.model(self.frames, actions)

        new_states = torch.argmax(new_states, dim=1)
        self.frames = torch.cat((self.frames[:, 3:], new_states.float() / 255), dim=1)

        new_states = (self.frames * 255).byte().detach().cpu()
        rewards = (
            (torch.argmax(rewards, dim=1).detach().cpu() - 1).numpy().astype("float")
        )

        rewards = {self.env.agents[i]: rewards[i] for i in range(len(self.env.agents))}
        done = {self.env.agents[i]: False for i in range(len(self.env.agents))}
        done["__all__"] = False
        return new_states, rewards, done, {}

    def reset(self):
        observation = self.env.reset()
        zero_frame = torch.zeros(observation.shape)
        return torch.stack([zero_frame, zero_frame, zero_frame, observation]).view(
            -1, *observation.shape[1:]
        )

    def get_initial_frame(self):
        return self.env.get_initial_frame()
