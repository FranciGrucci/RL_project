from actor_critic import Actor, Critic
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
import numpy as np
import time
from ornsteinuhlebeck import OrnsteinUhlenbeckNoise
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---- DDPG AGENT ----


class DDPG_Agent:
    def __init__(self, state_dim, action_dim, max_action, env, eval=False, gamma=0.99, tau=0.005, lr=0.001):
        # self.actor = Actor(learning_rate=lr, input_dims=state_dim, fc1_dims=400, fc2_dims=300, n_actions=action_dim, name="Actor", ckpt_dir="ckpt")
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim=state_dim,action_dim=action_dim,max_action=max_action).to(self.device)
        # self.actor_target = Actor(learning_rate=lr, input_dims=state_dim, fc1_dims=400, fc2_dims=300, n_actions=action_dim, name="Actor_target", ckpt_dir="ckpt")
        self.actor_target = Actor(state_dim=state_dim,action_dim=action_dim,max_action=max_action).to(self.device)
        #self.actor_target.eval()
        self.actor_target.load_state_dict(self.actor.state_dict())

        # self.critic = Critic(learning_rate=lr, input_dims=state_dim, fc1_dims=400, fc2_dims=300, n_actions=action_dim, name="Critic", ckpt_dir="ckpt")
        self.critic = Critic(state_dim=state_dim,action_dim=action_dim).to(self.device)
        # self.critic_target = Critic(learning_rate=lr, input_dims=state_dim, fc1_dims=400, fc2_dims=300, n_actions=action_dim, name="Critic_target", ckpt_dir="ckpt")
        self.critic_target = Critic(state_dim=state_dim,action_dim=action_dim).to(self.device)
        #self.critic_target.eval()
        self.eval = eval
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr)

        self.replay_buffer = ReplayBuffer(device=self.device)
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.noise = 0.1 #OrnsteinUhlenbeckNoise(action_dim=3)
        self.env = env
        self.max_mean_reward = 0
        self.rewards = 0
        self.update_loss = []
        self.reward_threshold = 300
        self.training_rewards = []
        self.mean_training_rewards = []
        self.actor_loss = []
        self.critic_loss = []
        self.window = 10
        self.step_count = 0
        
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
    
    def exponential_annealing_schedule(self,n, rate,start_value):
            return start_value * np.exp(-rate * n)

    def select_action(self, state, noise=0.1):
        self.actor.eval()
        with torch.no_grad():
            # if state.dim() == 1:  # Add batch dimension if state is a single image
            #     state = state.unsqueeze(0)  # Shape becomes (1, C, H, W)
            action = self.actor(state).detach().cpu().numpy()[0]
            #print(action)
            if not self.eval:
                action = action + np.random.normal(0, noise, size=action.shape)# self.noise.noise()   # Esplorazione
            # Limita le azioni
        self.actor.train()
        
        #return np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        return np.clip(action, [-3.0], [3.0])

    def handle_state_shape(self, s_0, device):
        if s_0.shape == torch.Size([3, 84, 96]):  # Ensures no further crops
            return s_0

        s_0 = torch.FloatTensor(s_0).detach()

        # Permute to change the order of dimensions
        # From (84, 3, 96) to (3, 96, 84)
        s_0 = s_0.permute(2, 1, 0)
        s_0 = s_0[:, :-12, :]

        s_0 = s_0.to(device)
        return s_0

    def take_step(self, mode='exploit'):

        if mode == 'explore':
            action = self.env.action_space.sample()

        else:

            # Assuming self.s_0 has shape 1x84x3x96
            #self.s_0 = self.handle_state_shape(self.s_0, self.device)
            #self.s_0 = torch.FloatTensor(self.s_0).detach().to(self.device)
            action = self.select_action(self.s_0, noise=self.noise)

        s_1, r, terminated, truncated, _ = self.env.step(action)
        #s_1 = self.handle_state_shape(s_1, self.device)

        done = terminated or truncated
        s_1 = torch.FloatTensor(s_1).detach().to(self.device)
        # put experience in the buffer
        self.replay_buffer.push(self.s_0, action, r, done, s_1)

        self.rewards += r

        self.s_0 = s_1.clone()

        if done:
            #self.noise.reset()
            self.s_0, _ = self.env.reset()
            #self.s_0 = self.handle_state_shape(self.s_0, self.device)
            self.s_0 = torch.FloatTensor(self.s_0).detach().to(self.device)
        return done

    def train(self, batch_size=32, n_episodes=10):
        self.actor.train()
        self.critic.train()
        #self.noise.reset()
        state, _ = self.env.reset()
        # for i in range(50):
        #     state,_,_,_,_ = self.env.step([0,0,0])
        #self.s_0 = self.handle_state_shape(state, self.device)
        self.s_0 = torch.FloatTensor(state).detach().to(self.device)
        print("Populating buffer")
        # Populate replay buffer
        while self.replay_buffer.burn_in_capacity() < 1:
            print("\rFull {:.2f}%\t\t".format(
                self.replay_buffer.burn_in_capacity()*100), end="")
            done = self.take_step(mode='explore')
            # if done:
            #     # for i in range(50):
            #     #     state,_,_,_,_ = self.env.step([0,0,0])
            #     #self.s_0 = self.handle_state_shape(state, self.device)
            #     self.s_0 = torch.FloatTensor(self.s_0).detach().to(self.device)
        print("\nStart training...")

        for episode in range(n_episodes):
            state, _ = self.env.reset()
            # for i in range(50):
            #     state,_,_,_,_ = self.env.step([0,0,0])
            #self.s_0 = self.handle_state_shape(state, self.device)
            self.s_0 = torch.FloatTensor(state).detach().to(self.device)

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
                    target_Q = rewards + (1 - dones) * self.gamma * target_Q
                
                self.critic_optimizer.zero_grad()

                # Optimize Critic
                current_Q = self.critic(
                    states, actions)

                critic_loss = F.mse_loss(current_Q, target_Q)
                critic_loss.backward()
                self.critic_optimizer.step()

                self.actor_optimizer.zero_grad()

                # Optimize Actor (maximize Q-value)
                actor_loss = -self.critic(states, self.actor(states)).mean()
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
                    if (episode % 10 == 0):  # Save checkpoint
                        self.noise = self.exponential_annealing_schedule(episode,1e-2,start_value=self.noise)
                        if self.noise <= 0.0001:
                            self.noise = 0.0001
                    if (episode % 20 == 0):  # Save checkpoint
                        print("Saving...")
                        self.save(filename="checkpoint.pth")
                    # if self.rewards > 2000:
                    #     self.training_rewards.append(2000)
                    # elif self.rewards > 1000:
                    #     self.training_rewards.append(1000)
                    # elif self.rewards > 500:
                    #     self.training_rewards.append(500)
                    # else:
                    self.training_rewards.append(self.rewards)
                    self.update_loss = []
                    mean_rewards = np.mean(self.training_rewards[-self.window:])
                    #mean_loss = np.mean(self.training_loss[-self.window:])
                    self.actor_loss.append(actor_loss.detach().cpu().item())   # Save loss
                    self.critic_loss.append(critic_loss.detach().cpu().item())
                    self.mean_training_rewards.append(mean_rewards)
                    if mean_rewards>self.max_mean_reward:
                        print("Saving...")
                        self.save(filename="best.pth")
                        self.max_mean_reward = mean_rewards
                    

                    print(
                        "\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f} \t\t".format(
                            episode, mean_rewards, self.rewards), end="")
                    
                    if mean_rewards >= self.reward_threshold:
                        #training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            episode))
                        self.save(filename="solved.pth")
                    if self.rewards>0:
                        print("<-----------")

                    
        self.plot_training_results()
        #self.plot_actor_loss()
        #self.plot_critic_loss()


        self.save(filename="final_ckeckpoint.pth")
        self.env.close()

    def evaluate(self, env):
        """
        Valuta un agente addestrato sull'ambiente CarRacing-v2.
        """
        self.load(filename="best.pth")

        total_reward = 0
        done = False
        state, _ = env.reset()
        #state = self.handle_state_shape(state, self.device)
        self.actor.eval()
        self.critic.eval()
        # for i in range(50):
        #     state,_,_,_,_ = env.step([0,0,0])
        #state = self.handle_state_shape(state, self.device)
        state = torch.FloatTensor(state).detach().to(self.device)

        with torch.no_grad():
            while True:
                
                action = self.select_action(state, noise=0.0)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                #state = self.handle_state_shape(next_state, self.device)
                state = torch.FloatTensor(next_state).detach().to(self.device)
                # if done:
                #     state, _ = env.reset()
                #     state = torch.FloatTensor(next_state).detach().to(self.device)
                #     print(total_reward)
                #     total_reward=0


        print(f" Reward = {total_reward}")
        #env.close()

    def plot_training_results(self):
        plt.figure(figsize=(18, 5))

        # Plot dei reward medi
        plt.subplot(1, 3, 1)
        plt.plot(self.mean_training_rewards)
        plt.title("Mean Training Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")
       

        # Plot delle loss
  
        plt.subplot(1, 3, 2)
        plt.plot(self.actor_loss, label="Actor Loss")
        
        plt.title("Actor Loss During Training")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()

    
        plt.subplot(1, 3, 3)
        
        plt.plot(self.critic_loss, label="Critic Loss")
        plt.title("Critic Loss During Training")
        plt.xlabel("Training Steps")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plt.savefig('mean_training_rewards.png')
        

        plt.show()