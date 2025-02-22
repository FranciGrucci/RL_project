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
    """DDPG agent class
    """

    def __init__(self, env, state_dim, action_dim, max_action, device, replay_buffer, noise, eval=False, gamma=0.99, tau=0.005, actor_lr=0.001, critic_lr=0.001):
        """DDPG agent init function

        Args:
            env : Gymnasium environment
            state_dim (int): State dimension
            action_dim (int): number of actions
            max_action (float): maximum allowed action value
            device: "cuda" or "cpu"
            replay_buffer: Replay buffer object
            noise: Noise object
            eval (bool, optional): True when the agent is in evaluation mode. Defaults to False.
            gamma (float, optional): discount factor. Defaults to 0.99.
            tau (float, optional): Soft update tau. Defaults to 0.005.
            actor_lr (float, optional): Actor learning rate. Defaults to 0.001.
            critic_lr (float, optional): Critic learning rate. Defaults to 0.001.
        """
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

        self.rewards = 0
        self.training_rewards = []
        self.mean_training_rewards = []
        self.mean_rewards = 0
        self.max_mean_reward = 0

        self.actor_loss = []
        self.critic_loss = []

        self.env = env
        self.eval = eval

    def save(self, filename="ddpg_checkpoint.pth"):
        """Load checkpoint

        Args:
            filename (str, optional): checkpoint filename. Defaults to "ddpg_checkpoint.pth".
        """
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict()
        }
        torch.save(checkpoint, filename)
        print(f"Modello salvato in {filename}")

    def load(self, filename="ddpg_checkpoint.pth"):
        """Load checkpoint.

        Args:
            filename (str, optional): checkpoint filename. Defaults to "ddpg_checkpoint.pth".
        """
        checkpoint = torch.load(filename, map_location="cpu")
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.actor_optimizer.load_state_dict(
            checkpoint["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(
            checkpoint["critic_optimizer_state_dict"])
        print(f"Modello caricato da {filename}")

    def select_action(self, state):
        """Select action for current state

        Args:
            state (Torch): State tensor

        Returns:
            Array: Action array + noise if self.eval = True. Action array otherwise
        """
        self.actor.eval()

        with torch.no_grad():
            if state.dim() == 1:  # Add batch dimension if single sample
                state = state.unsqueeze(0)
            action = self.actor(state).detach().cpu().numpy()[0]
            if not self.eval:
                action = action + self.noise.sample()

        self.actor.train()

        # Clip to max action range
        return np.clip(action, -self.max_action * np.ones(self.action_dim), self.max_action * np.ones(self.action_dim))

    def take_step(self, mode='exploit'):
        """Step wrapper

        Args:
            mode (str, optional): {exploit, explore}. If "explore", the environment action space is sampled. Otherwise, the action is chosen accordingly to select_action(). Defaults to 'exploit'.

        Returns:
            bool: True if done. False otherwise
        """

        if mode == 'explore':
            action = self.env.action_space.sample()

        else:

            action = self.select_action(self.s_0)

        s_1, r, terminated, truncated, _ = self.env.step(action)
        s_1 = torch.FloatTensor(s_1).detach().to(self.device)

        done = terminated or truncated

        # put experience in the buffer
        self.replay_buffer.push(self.s_0, action, r, done, s_1)

        self.rewards += r

        self.s_0 = s_1.clone()

        if done:
            self.noise.on_next_episode()
            self.s_0, _ = self.env.reset()
            self.s_0 = torch.FloatTensor(self.s_0).detach().to(self.device)
        return done

    def train(self, batch_size=32, checkpoint_frequency=50,window = 20,mean_reward_threshold=2000):
        """Train loop

        Args:
            batch_size (int, optional): Batch size. Defaults to 32.
            checkpoint_frequency (int, optional): Number of episodes after which the checkpoint is created. Defaults to 50.
        """
        self.actor.train()
        self.critic.train()

        self.noise.on_next_episode()  # Reset the noise, useful with Ornstein-Uhlenbeck
        state, _ = self.env.reset()
        self.s_0 = torch.FloatTensor(state).detach().to(self.device)
        print("Populating buffer")

        # Populate replay buffer
        while self.replay_buffer.burn_in_capacity() < 1:

            print("\rFull {:.2f}%\t\t".format(
                self.replay_buffer.burn_in_capacity()*100), end="")
            done = self.take_step(mode='explore')

        print("\nStart training...")
        
        train = True  # Flag
        episode = 0
        # Train until objective is not reached (Based on manual abortion)
        while not self.mean_rewards >= mean_reward_threshold:
            self.noise.on_next_episode()
            state, _ = self.env.reset()
            self.s_0 = torch.FloatTensor(state).detach().to(self.device)
            self.rewards = 0
            done = False

            while not done:  # Episode loop
                done = self.take_step(mode="exploit")

                states, actions, rewards, dones, next_states = self.replay_buffer.sample_batch(
                    batch_size)

                # Compute target Q-value
                with torch.no_grad():

                    next_actions = self.actor_target(next_states)
                    target_Q = self.critic_target(next_states, next_actions)
                    target_Q = rewards + (1 - dones) * self.gamma * target_Q

                actor_loss, critic_loss = self.replay_buffer.compute_loss(
                    actor=self.actor, actor_optimizer=self.actor_optimizer, critic=self.critic, critic_optimizer=self.critic_optimizer, states=states, actions=actions, target_Q=target_Q)

                # Target networks soft updates
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(
                        self.tau * param.data + (1 - self.tau) * target_param.data)

                if done:  # Episode is over

                    if (episode % checkpoint_frequency == 0):  # Save checkpoint
                        print("Saving...")
                        self.save(filename="checkpoint.pth")
                        self.plot_training_results(
                            filename="checkpoint", show=False)

                    # Collect data for stats and plots
                    self.training_rewards.append(self.rewards)
                    self.mean_rewards = np.mean(
                        self.training_rewards[-window:])
                    self.actor_loss.append(
                        actor_loss.detach().cpu().item())
                    self.critic_loss.append(critic_loss.detach().cpu().item())
                    self.mean_training_rewards.append(self.mean_rewards)

                    if self.mean_rewards > self.max_mean_reward:  # New max mean reward reached
                        print("Saving...")
                        self.save(filename="best.pth")
                        self.max_mean_reward = self.mean_rewards

                    print(
                        "\rEpisode {:d} Mean Rewards {:.2f}  Episode reward = {:.2f} \t\t".format(
                            episode, self.mean_rewards, self.rewards), end="")

                    if self.mean_rewards >= mean_reward_threshold:  # Objective reached
                        print('\nEnvironment solved in {} episodes!'.format(
                            episode))
                        self.save(filename="solved.pth")
                        self.plot_training_results(
                            filename="solved", show=True)
                        train = False  # Signal that train is over
                    else:  # Move to next episode
                        episode += 1
                        self.replay_buffer.on_next_episode()

            # Exit the Train loop when objective is reached (hence train is false)
            if not train:
                break

        self.env.close()

    def evaluate(self, env, checkpoint_path):
        """Agent evaluation function

        Args:
            env: Gymnasium environment
            checkpoint_path (str): Checkpoint path
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

                action = self.select_action(state)
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

    def plot_training_results(self, filename='mean_training_rewards', show=False):
        """Plot training results and save on png file.

        Args:
            filename (str, optional): Image file name. Defaults to 'mean_training_rewards'.
            show (bool, optional): If true, show the plot. Defaults to False.
        """
        plt.figure(figsize=(18, 5))

        # Mean reward plot
        plt.subplot(1, 3, 1)
        plt.plot(self.mean_training_rewards)
        plt.title("Mean Training Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Reward")

        # Loss plot

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
