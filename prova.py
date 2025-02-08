import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import random
import os
import pathlib

# ---- ACTOR NETWORK ----
class Actor(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions, name, ckpt_dir= 'ckpt'):
        super(Actor, self).__init__()
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cpu')
        self.to(self.device)

        if not os.path.exists(ckpt_dir):
            pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x)) # Azioni in [-1, 1]
        return x
    # def save_checkpoint(self):
    #     torch.save(self.state_dict(), self.checkpoint_file)

    # def load_checkpoint(self):
    #     self.load_state_dict(torch.load(self.checkpoint_file))

# ---- CRITIC NETWORK ----
class Critic(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions, name, ckpt_dir= 'ckpt'):
        super(Critic, self).__init__()
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg')
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device('cpu')
        self.to(self.device)


    # def forward(self, state, action):
    #     state_value = F.relu(self.bn1(self.fc1(state)))
    #     state_value = F.relu(self.bn2(self.fc2(state_value)))
    #     action_value = F.relu(self.action_value(action))
    #     state_action_value = F.relu(torch.add(state_value, action_value))
    #     return self.q(state_action_value)
    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        
        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        return self.q(state_action_value)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

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
        self.actor = Actor(learning_rate=lr, input_dims=state_dim, fc1_dims=400, fc2_dims=300, n_actions=action_dim, name="Actor", ckpt_dir="ckpt")
        self.actor_target = Actor(learning_rate=lr, input_dims=state_dim, fc1_dims=400, fc2_dims=300, n_actions=action_dim, name="Actor_target", ckpt_dir="ckpt")
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(learning_rate=lr, input_dims=state_dim, fc1_dims=400, fc2_dims=300, n_actions=action_dim, name="Critic", ckpt_dir="ckpt")
        self.critic_target = Critic(learning_rate=lr, input_dims=state_dim, fc1_dims=400, fc2_dims=300, n_actions=action_dim, name="Critic_target", ckpt_dir="ckpt")
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer()
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action

    def save(self, filename="ddpg_checkpoint.pth"):
        """Salva i parametri delle reti dell'agente."""
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict()
        }
        torch.save(checkpoint, filename)
        print(f"Modello salvato in {filename}")

    def load(self, filename="ddpg_checkpoint.pth"):
        """Carica i parametri delle reti dell'agente."""
        checkpoint = torch.load(filename)
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer_state_dict"])
        print(f"Modello caricato da {filename}")


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

