from actor_critic import Actor,Critic
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
import numpy as np



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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        state = torch.tensor(state, dtype=torch.float32,device=self.device).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        action = action + np.random.normal(0, noise, size=action.shape)  # Esplorazione
        return action #np.clip(action, -self.max_action, self.max_action)  # Limita le azioni

    def train(self, batch_size=64):
        if self.replay_buffer.size() < batch_size:
            return

        states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch(batch_size)

        # Compute target Q-value
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, next_actions)
            #print(type(target_Q))
            #print("ELLE",(rewards.get_device(), dones.get_device(),target_Q.get_device()))
            target_Q = rewards.to(self.device) + (1 - dones.to(self.device)) * self.gamma * target_Q

        # Optimize Critic
        current_Q = self.critic(states, torch.tensor(actions,device=self.device))
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

