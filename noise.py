import numpy as np


class OrnsteinUhlenbeckNoise:
    """Ornstein-Uhlenbeck noise class
    """
    def __init__(self, action_dim, mu=0, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.action_dim = action_dim
        self.theta = theta
        self.mu = np.full(action_dim, mu)
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        """reset state
        """
        self.state = self.x0 if self.x0 is not None else np.zeros(
            self.action_dim)

    def sample(self):
        """Draw random samples following  Ornstein-Uhlenbeck noise. Sample will have dimension = action_dim

        Returns:
            array: random sample of dimension action_dim
        """
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * \
            np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state += dx
        return self.state

    def on_next_episode(self):
        """Action to execute when next episode occurs
        """
        self.reset()

    def __str__(self):
        return f"Ornstein-Uhlenbeck Noise"


class GaussianNoise:
    """Ornstein-Uhlenbeck noise class
    """
    def __init__(self, mean, std, action_dim):
        self.mean = mean
        self.std = std
        self.shape = action_dim

    def sample(self):
        """Draw random samples following Gaussuan noise. Sample will have dimension = action_dim

        Returns:
            array: random sample of dimension action_dim
        """
        return np.random.normal(self.mean, self.std, size=self.shape)

    def on_next_episode(self):
        """Action to execute when next episode occurs
        """
        pass

    def __str__(self):
        return f"Gaussian Noise"
