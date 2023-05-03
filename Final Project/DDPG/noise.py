import numpy as np


class OrnsteinUhlenbeckNoise:
    def __init__(self, n_actions, mu=0.0, theta=0.3, sigma=0.4):
        self.state = None
        self.mu = mu * np.ones(n_actions)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=self.mu.shape)
        self.state = x + dx
        return self.state
