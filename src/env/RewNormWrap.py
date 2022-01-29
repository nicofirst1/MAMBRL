# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
import numpy as np

from src.common import mas_dict2tensor


class RunningMeanStd:
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class NormalizeReward:
    def __init__(
            self,
            env,
            gamma=0.99,
            epsilon=1e-8,
    ):
        self.env=env
        self.num_agents = getattr(env, "num_agents", 1)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_agents)
        self.gamma = gamma
        self.epsilon = epsilon

    def __getattr__(self, item):
        if item in self.__dict__.keys():
            return self.__dict__[item]

        return getattr(self.env, item)

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)

        rews= mas_dict2tensor(rews).numpy()
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        self.returns[dones["__all__"]] = 0.0

        rews={self.env.agents[idx]:rews[idx] for idx in range(self.num_agents)}

        return obs, rews, dones, infos

    def normalize(self, rews):
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)