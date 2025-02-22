from actor_critic import Actor, Critic
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import matplotlib.pyplot as plt

# ---- DDPG AGENT ----


class DDPG_Agent:
    def __init__(self, env, state_dim, action_dim, max_action,device, replay_buffer, eval=False, noise=0.1, gamma=0.99, tau=0.005, actor_lr=0.001, critic_lr=0.001, memory_size=50000, burn_in=40000):

        self.device = device
        ############################## ACTOR #####################################################
        self.actor = Actor(state_dim=state_dim, action_dim=action_dim,
                           max_action=max_action).to(self.device)
        
        self.actor_optimizer = optim.AdamW(
            self.actor.parameters(), lr=actor_lr)

        self.actor_target = Actor(
            state_dim=state_dim, action_dim=action_dim, max_action=max_action).to(self.device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())

        #########################################################################################

        ############################ CRITIC #####################################################

        self.critic = Critic(state_dim=state_dim,
                             action_dim=action_dim).to(self.device)

        self.critic_optimizer = optim.AdamW(
            self.critic.parameters(), lr=critic_lr)

        self.critic_target = Critic(
            state_dim=state_dim, action_dim=action_dim).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        ############################################################################################

        self.replay_buffer = replay_buffer

        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.action_dim = action_dim
        self.noise = noise
        
        self.reward_threshold = 2000
        self.rewards = 0
        self.training_rewards = []
        self.mean_training_rewards = []
        self.mean_rewards = 0
        self.max_mean_reward = 0

        self.update_loss = []
        self.actor_loss = []
        self.critic_loss = []

        self.learning_rates = []
        self.episode_numbers = []

        self.window = 20

        self.env = env
        self.eval = eval

    def replay_buffer_exponential_annealing_schedule(self, n, rate, start_value=0.4):
        # from start_value to 1
        return 1 - (1-start_value)*np.exp(-rate * (n/100))

    def exponential_annealing_schedule(self, n, rate, start_value=0.4):
        return start_value * np.exp(-rate * n)  # from start_value to 0

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
        checkpoint = torch.load(filename, map_location="cpu")
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state_dict"])
        print(f"Modello caricato da {filename}")

    def select_action(self, state, noise=0.1):
        self.actor.eval()

        with torch.no_grad():
            if state.dim() == 1:  # Add batch dimension if single sample
                state = state.unsqueeze(0)  # Shape becomes (1, C, H, W)
            action = self.actor(state).detach().cpu().numpy()[0]
            if not self.eval:
                if type(self.noise) == float:
                    # self.noise.noise()   # Esplorazione
                    action = action + \
                        np.random.normal(0, noise, size=action.shape)
                else:
                    action = action + self.noise.sample()

        self.actor.train()

        # Clip to max action range
        return np.clip(action, -self.max_action * np.ones(self.action_dim), self.max_action * np.ones(self.action_dim))

    def take_step(self, mode='exploit'):

        if mode == 'explore':
            action = self.env.action_space.sample()

        else:

            action = self.select_action(self.s_0, noise=self.noise)

        s_1, r, terminated, truncated, _ = self.env.step(action)
        s_1 = torch.FloatTensor(s_1).detach().to(self.device)

        done = terminated or truncated

        # put experience in the buffer
        self.replay_buffer.push(self.s_0, action, r, done, s_1)

        self.rewards += r

        self.s_0 = s_1.clone()

        if done:
            self.noise.reset()
            self.s_0, _ = self.env.reset()
            self.s_0 = torch.FloatTensor(self.s_0).detach().to(self.device)
        return done

    def train(self, batch_size=32,checkpoint_frequency = 50):
        # = 0.0001  # Learning rate iniziale
        # final_lr = 0.00001   # Learning rate minimo
        # decay_rate = (final_lr / initial_lr) ** (1 / n_episodes)
        self.actor.train()
        self.critic.train()
        self.noise.reset()
        state, _ = self.env.reset()
        self.s_0 = torch.FloatTensor(state).detach().to(self.device)
        print("Populating buffer")

        # Populate replay buffer
        while self.replay_buffer.burn_in_capacity() < 1:

            print("\rFull {:.2f}%\t\t".format(
                self.replay_buffer.burn_in_capacity()*100), end="")
            done = self.take_step(mode='explore')

        print("\nStart training...")
        train = True
        episode = 0
        while not self.mean_rewards >= self.reward_threshold:
            self.noise.reset()
            state, _ = self.env.reset()
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

                actor_loss,critic_loss = self.replay_buffer.compute_loss(actor = self.actor,actor_optimizer=self.actor_optimizer,critic = self.critic,critic_optimizer=self.critic_optimizer,states = states,actions = actions, target_Q =target_Q)

                # Soft update delle reti target
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

                if done:

                    if (episode % checkpoint_frequency == 0):  # Save checkpoint
                        print("Saving...")
                        self.save(filename="checkpoint.pth")
                        self.plot_training_results(
                            filename="checkpoint", show=False)

                    self.training_rewards.append(self.rewards)
                    self.update_loss = []
                    self.mean_rewards = np.mean(
                        self.training_rewards[-self.window:])
                    self.actor_loss.append(
                        actor_loss.detach().cpu().item())   # Save loss
                    self.critic_loss.append(critic_loss.detach().cpu().item())
                    self.mean_training_rewards.append(self.mean_rewards)

                    if self.mean_rewards > self.max_mean_reward:
                        print("Saving...")
                        self.save(filename="best.pth")
                        self.max_mean_reward = self.mean_rewards

                    print(
                        "\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f} \t\t".format(
                            episode, self.mean_rewards, self.rewards), end="")

                    if self.mean_rewards >= self.reward_threshold:
                        # training = False
                        print('\nEnvironment solved in {} episodes!'.format(
                            episode))
                        self.save(filename="solved.pth")
                        self.plot_training_results(
                            filename="solved", show=True)
                        train = False
                    episode += 1
                    self.replay_buffer.on_next_episode()

            if not train:
                break
        if train:
            self.plot_training_results()
            
        self.save(filename="final_ckeckpoint.pth")
        self.env.close()

    def evaluate(self, env, checkpoint_path):
        """
        Valuta un agente addestrato sull'ambiente CarRacing-v2.
        """
        self.load(filename=checkpoint_path)

        total_reward = 0
        done = False
        state, _ = env.reset()
        self.actor.eval()
        self.critic.eval()

        state = torch.FloatTensor(state).detach().to(self.device)

        with torch.no_grad():
            while True:

                action = self.select_action(state, noise=0.0)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = torch.FloatTensor(next_state).detach().to(self.device)
                if done:
                    state, _ = env.reset()
                    state = torch.FloatTensor(
                        next_state).detach().to(self.device)
                    print(total_reward)
                    total_reward = 0
                    # time.sleep(10)
                    loaded = False
                    while not loaded:
                        try:
                            self.load(filename=checkpoint_path)
                            loaded = True
                            break  # Exit loop if successful
                        except Exception as e:
                            print("Loading failed... trying again in 5 seconds")
                            time.sleep(5)

        print(f" Reward = {total_reward}")
        env.close()

    def plot_learning_rate(self, filename="learning_rate_curve.png", show=False):
        plt.figure(figsize=(10, 6))
        plt.plot(self.episode_numbers, self.learning_rates,
                 label='Learning Rate', color='red')
        plt.title('Adaptive Learning Rate Decay')
        plt.xlabel('Episode')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid()
        plt.savefig(filename)  # Salva la curva
        plt.show()

    def plot_training_results(self, filename='mean_training_rewards', show=False):
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
        plt.savefig(filename + '.png')
        if show:

            plt.show()
        plt.close()
