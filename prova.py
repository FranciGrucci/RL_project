import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import random

# ---- ACTOR NETWORK ----
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # tanh perché le azioni sono tra -1 e 1
        return x * self.max_action

# ---- CRITIC NETWORK ----
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)  # Concatena stato e azione
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Q-value

# ---- REPLAY BUFFER ----
class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(np.array(actions), dtype=torch.float32),
                torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1))

    def size(self):
        return len(self.buffer)

# ---- DDPG AGENT ----
class DDPG_Agent:
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, lr=0.001):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

    def select_action(self, state, noise=0.1):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = self.actor(state).detach().numpy()[0]
        action = action + np.random.normal(0, noise, size=action.shape)  # Esplorazione
        return np.clip(action, -self.max_action, self.max_action)  # Limita le azioni

    def train(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Compute target Q-value
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        # Optimize Critic
        current_Q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_Q, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Optimize Actor (maximize Q-value)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update delle reti target
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# ---- ESPERIENZA NELL'AMBIENTE ----
env = gym.make("CarRacing-v2", continuous=True)
state_dim = env.observation_space.shape[0] * env.observation_space.shape[1] * env.observation_space.shape[2]  # Flatten immagine
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

agent = DDPG_Agent(state_dim, action_dim, max_action)

# Addestramento
num_episodes = 500
for episode in range(num_episodes):
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32).flatten()  # Flatten immagine
    episode_reward = 0

    for step in range(1000):  # Limite max passi
        action = agent.select_action(state)
        next_state, reward, done, _, _ = env.step(action)
        next_state = np.array(next_state, dtype=np.float32).flatten()

        agent.replay_buffer.push(state, action, reward, next_state, done)
        agent.train()

        state = next_state
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode}, Reward: {episode_reward}")

env.close()
