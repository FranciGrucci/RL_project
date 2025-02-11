import torch
import numpy as np
import random
from collections import deque
import time

# ---- REPLAY BUFFER ----
class ReplayBuffer:
    def __init__(self, device,max_size=25000):
        self.buffer = deque(maxlen=max_size)
        self.burn_in = 10000
        self.device = device

    def push(self, state, action, reward, done,next_state):
        
        self.buffer.append((state, action, reward,done, next_state))
        
    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards,dones, next_states = zip(*batch)
        
        rewards = torch.FloatTensor(rewards).reshape(-1, 1).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        dones = torch.IntTensor(dones).reshape(-1, 1).to(self.device)
        states = torch.stack(states)
        next_states = torch.stack(next_states)
                
        return states, actions, rewards,dones, next_states

    def size(self):
        return len(self.buffer)
    
    def burn_in_capacity(self):
        return len(self.buffer) / self.burn_in