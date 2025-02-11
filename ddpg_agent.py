from actor_critic import Actor, Critic
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
import numpy as np
import time
from tqdm import tqdm
# ---- DDPG AGENT ----
class DDPG_Agent:
    def __init__(self, state_dim, action_dim, max_action, env, gamma=0.99, tau=0.005, lr=0.001):
        # self.actor = Actor(learning_rate=lr, input_dims=state_dim, fc1_dims=400, fc2_dims=300, n_actions=action_dim, name="Actor", ckpt_dir="ckpt")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(action_dim).to(self.device)

        # self.actor_target = Actor(learning_rate=lr, input_dims=state_dim, fc1_dims=400, fc2_dims=300, n_actions=action_dim, name="Actor_target", ckpt_dir="ckpt")
        self.actor_target = Actor(action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # self.critic = Critic(learning_rate=lr, input_dims=state_dim, fc1_dims=400, fc2_dims=300, n_actions=action_dim, name="Critic", ckpt_dir="ckpt")
        self.critic = Critic(action_dim).to(self.device)
        # self.critic_target = Critic(learning_rate=lr, input_dims=state_dim, fc1_dims=400, fc2_dims=300, n_actions=action_dim, name="Critic_target", ckpt_dir="ckpt")
        self.critic_target = Critic(action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(device=self.device)
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        
        self.env = env
        self.rewards = 0


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
        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state_dict"])
        print(f"Modello caricato da {filename}")


    def select_action(self, state, noise=0.1):
        
        action = self.actor(state).detach().cpu().numpy()[0]
        action = action + np.random.normal(0, noise, size=action.shape)  # Esplorazione
        return np.clip(action, -self.max_action, self.max_action)  # Limita le azioni


    def handle_state_shape(self, s_0, device):
        if s_0.shape == torch.Size([3, 84, 96]):  # Ensures no further crops
            return s_0

        s_0 = torch.FloatTensor(s_0)

        # Permute to change the order of dimensions
        # From (84, 3, 96) to (3, 96, 84)
        s_0 = s_0.permute(2, 1, 0)
        s_0 = s_0[:, :-12, :]

        s_0 = s_0.to(device)
        return s_0

    def take_step(self, mode='exploit'):
        # choose action with epsilon greedy
        if mode == 'explore':
            action = self.env.action_space.sample()

        else:
            
            # Assuming self.s_0 has shape 1x84x3x96
            self.s_0 = self.handle_state_shape(self.s_0, self.device)

            action = self.select_action(self.s_0, noise=0.1)
      

        s_1, r, terminated, truncated, _ = self.env.step(action)
        s_1 = self.handle_state_shape(s_1, self.device)

        done = terminated or truncated

        # put experience in the buffer
        self.replay_buffer.push(self.s_0, action, r, done, s_1)

        self.rewards += r

        self.s_0 = s_1.detach().clone()

        # self.step_count += 1

        if done:
            self.s_0, _ = self.env.reset()
            self.s_0 = self.handle_state_shape(self.s_0, self.device)
        return done

    def train(self, batch_size=32, n_episodes=10):
        self.s_0, _ = self.env.reset()
        self.s_0 = self.handle_state_shape(self.s_0, self.device)
        print("Populating buffer")
        # Populate replay buffer
        while self.replay_buffer.burn_in_capacity() < 1:
            print("\rFull {:2f}%\t\t".format(
                self.replay_buffer.burn_in_capacity()*100), end="")
            self.take_step(mode='explore')

        print("\nStart training...")

        for episode in tqdm(range(n_episodes)):
            self.s_0, _ = self.env.reset()
            self.s_0 = self.handle_state_shape(self.s_0, self.device)
            self.rewards = 0
            done = False
            while not done:
                done = self.take_step(mode="exploit")

                states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch(
                    batch_size)

                # Compute target Q-value
                with torch.no_grad():
                    next_actions = self.actor_target(next_states)
                    target_Q = self.critic_target(next_states, next_actions)
                    #print(target_Q.shape)
                    #time.sleep(5)
                    # print(type(target_Q))
                    # print("ELLE",(rewards.get_device(), dones.get_device(),target_Q.get_device()))
                    target_Q = rewards.to(
                        self.device) + (1 - dones.to(self.device)) * self.gamma * target_Q

                # Optimize Critic
                current_Q = self.critic(
                    states, actions)
                
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
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

                if done:
                    print(
                        f"Episodio {episode + 1}/{n_episodes}, Reward: {self.rewards}")
        self.save()
        self.env.close()
    
    def evaluate(self,env):
        
        """
        Valuta un agente addestrato sull'ambiente CarRacing-v2.
        """
        #agent = DDPG_Agent()
        self.load()

        
        #rewards = []

    
        total_reward = 0
        done = False
        state, _ = env.reset()
        #state = np.array(state, dtype=np.float32).flatten()
        state = self.handle_state_shape(state,self.device)
        while not done:
            action = self.select_action(state, noise=0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.handle_state_shape(next_state,self.device)
            #state = np.array(next_state, dtype=np.float32).flatten()

        #rewards.append(total_reward)
        print(f" Reward = {total_reward}")

        
    
        env.close()
