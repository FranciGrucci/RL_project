import numpy as np
# class OrnsteinUhlenbeckNoise:
#     def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
#         self.action_dim = action_dim
#         self.mu = mu
#         self.theta = theta
#         self.sigma = sigma
#         self.state = np.ones(self.action_dim) * self.mu

#     def reset(self):
#         self.state = np.ones(self.action_dim) * self.mu

#     def noise(self):
#         dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
#         self.state += dx
#         return self.state

class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, mu=0, sigma=0.2, theta=0.15, dt=1e-2, x0=None):
        self.action_dim = action_dim
        self.theta = theta
        self.mu = np.full(action_dim, mu)  # Mu deve essere un array della dimensione dell'azione
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.state = self.x0 if self.x0 is not None else np.zeros(self.action_dim)

    def noise(self):
        dx = self.theta * (self.mu - self.state) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        self.state += dx
        return self.state

    def __repr__(self):
        return f"OrnsteinUhlenbeckNoise(mu={self.mu}, sigma={self.sigma})"
