import numpy as np
from collections import namedtuple, deque
import torch


class Experience_replay_buffer:

    def __init__(self, device, memory_size=50000, burn_in=40000, alpha=1, beta=0):

        self.memory_size = memory_size
        self.burn_in = burn_in
        self.Buffer = namedtuple('Buffer',
                                 field_names=['state', 'action', 'reward', 'done', 'next_state'])
        self.replay_memory = np.empty(self.memory_size, dtype=[(
            "priority", np.float32), ("experience", self.Buffer)])  # deque(maxlen=memory_size)

        self.priorities = np.array([])
        self.priorities_prob = np.array([])
        self.alpha = alpha
        self.beta = beta
        self.sampled_priorities = np.array([])
        self._buffer_length = 0  # current number of prioritized experience tuples in buffer
        self.device = device

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

    def sum_scaled_priorities(self, scaled_priorities):
        return np.sum(scaled_priorities)

    def compute_probability(self):
        scaled_priorities = (
            self.replay_memory[:self._buffer_length]["priority"])

        self.priorities_prob = (scaled_priorities**self.alpha) / \
            np.sum(scaled_priorities**self.alpha)
        return self.priorities_prob
