import torch
import numpy as np
import random
from collections import namedtuple, deque
import time
from abc import ABC, abstractmethod
import torch.nn.functional as F


class ReplayBufferInterface(ABC):
    """Interface for Replay Buffers (Standard & PER)."""

    @abstractmethod
    def push(self, experience, priority=None):
        """Add experience to the buffer. Priority is optional."""
        pass

    @abstractmethod
    def sample_batch(self, batch_size):
        """Sample a batch of experiences."""
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def burn_in_capacity(self):
        pass

    @abstractmethod
    def compute_loss(self):
        "Compute loss accordingly to Replay Buffer type"
        pass

    @abstractmethod
    def on_next_episode(self):
        pass

# ---- REPLAY BUFFER ----


class ReplayBuffer(ReplayBufferInterface):
    def __init__(self, device, max_size=1000000, burn_in=90000):
        self.buffer = deque(maxlen=max_size)
        self.burn_in = burn_in
        self.device = device
        self.episode = 0

    def push(self, state, action, reward, done, next_state):

        self.buffer.append((state, action, reward, done, next_state))

    def sample_batch(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, dones, next_states = zip(*batch)

        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        dones = torch.FloatTensor(dones).reshape(-1, 1).to(self.device)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        return states, actions, rewards, dones, next_states

    def size(self):
        return len(self.buffer)

    def burn_in_capacity(self):
        return len(self.buffer) / self.burn_in

    def on_next_episode(self):
        return

    def compute_loss(self, actor, actor_optimizer, critic, critic_optimizer, states, actions, target_Q):

        critic_optimizer.zero_grad()

        # Optimize Critic
        current_Q = critic(
            states, actions)

        critic_loss = F.mse_loss(current_Q, target_Q)
        critic_loss.backward()
        critic_optimizer.step()

        actor_optimizer.zero_grad()

        # Optimize Actor (maximize Q-value)
        actor_loss = -critic(states, actor(states)).mean()
        actor_loss.backward()
        actor_optimizer.step()
        return actor_loss, critic_loss

    def __str__(self):
        return f"Replay Buffer"


class Experience_replay_buffer(ReplayBufferInterface):

    def __init__(self, device, memory_size=1000000, burn_in=90000, alpha=1, beta=0, beta_update_frequency=100):

        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer',
                                 field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = np.empty(self.memory_size, dtype=[(
            "priority", np.float32), ("experience", self.Buffer)])  # deque(maxlen=memory_size)

        self.priorities = np.array([])
        self.priorities_prob = np.array([])
        self.alpha = alpha
        self.beta0 = beta
        self.beta = beta
        self.beta_update_frequency = beta_update_frequency
        self.sampled_priorities = np.array([])
        self._buffer_length = 0  # current number of prioritized experience tuples in buffer
        self.device = device
        self.episode = 0

    def sample_batch(self, batch_size=32):

        samples = np.random.choice(np.arange((self.replay_memory[:self._buffer_length]["priority"]).size), batch_size,
                                   replace=True, p=self.compute_probability())
        self.sampled_priorities = samples

        experiences = self.replay_memory["experience"][samples]

        states, actions, rewards, dones, next_states = zip(*experiences)

        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        dones = torch.FloatTensor(dones).reshape(-1, 1).to(self.device)
        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)

        # print(experiences[0])

        return states, actions, rewards, dones, next_states

    def push(self, s_0, a, r, d, s_1):
        priority = 1.0 if self._buffer_length == 0 else self.replay_memory["priority"].max(
        )
        if self._buffer_length == self.memory_size:
            if priority > self.replay_memory["priority"].min():
                idx = self.replay_memory["priority"].argmin()
                self.replay_memory[idx] = (
                    priority, self.Buffer(s_0, a, r, d, s_1))
            else:
                pass  # low priority experiences should not be included in buffer
        else:
            self.replay_memory[self._buffer_length] = (
                priority, self.Buffer(s_0, a, r, d, s_1))
            self._buffer_length += 1

    def burn_in_capacity(self):
        # print(self._buffer_length)
        return self._buffer_length / self.burn_in

    def capacity(self):
        return len(self.replay_memory) / self.memory_size

    def size(self):
        return self._buffer_length

    def sum_scaled_priorities(self, scaled_priorities):
        return np.sum(scaled_priorities)

    def compute_probability(self):
        scaled_priorities = (
            self.replay_memory[:self._buffer_length]["priority"])

        self.priorities_prob = (scaled_priorities**self.alpha) / \
            np.sum(scaled_priorities**self.alpha)
        return self.priorities_prob

    def compute_weight(self):
        is_weights = self.replay_memory["priority"][self.sampled_priorities]
        is_weights *= self._buffer_length
        is_weights = ((is_weights)**(-self.beta))
        is_weights /= is_weights.max()
        return torch.Tensor(is_weights).view(-1).to(self.device)

    def replay_buffer_exponential_annealing_schedule(self, n, rate, start_value=0.4):
        # from start_value to 1
        return 1 - (1-start_value)*np.exp(-rate * (n/self.beta_update_frequency))

    def compute_loss(self, actor, actor_optimizer, critic, critic_optimizer, states, actions, target_Q):

        critic_optimizer.zero_grad()
        # # Optimize Critic
        current_Q = critic(
            states, actions)
        is_weights = self.compute_weight()

        # Optimize Critic
        critic_loss = (is_weights * F.mse_loss(current_Q,
                                               target_Q, reduction='none')).mean()

        critic_loss.backward()
        critic_optimizer.step()

        actor_optimizer.zero_grad()

        # Optimize Actor (maximize Q-value)
        actor_loss = -critic(states, actor(states)).mean()
        actor_loss.backward()
        actor_optimizer.step()

        self.replay_memory["priority"][self.sampled_priorities] = (
            target_Q-current_Q).abs().cpu().detach().numpy().flatten() + 1e-6

        return actor_loss, critic_loss

    def on_next_episode(self):
        if (self.episode % self.beta_update_frequency == 0 and self.episode != 0):  # Save checkpoint
            self.beta = self.replay_buffer_exponential_annealing_schedule(
                n=self.episode, rate=1e-2, start_value=self.beta0)
            print(f"Beta current value: {self.beta}")
        self.episode += 1

    def __str__(self):
        return f"Replay Buffer with PER"
