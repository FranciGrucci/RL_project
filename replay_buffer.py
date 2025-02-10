import torch
import numpy as np
import random
from collections import deque
import time
# ---- REPLAY BUFFER ----
class ReplayBuffer:
    def __init__(self, max_size=5000):
        self.buffer = deque(maxlen=max_size)
        self.burn_in = 1000

    def push(self, state, action, reward, done,next_state):
        #print("PRE",next_state)
        #time.sleep(1)
        self.buffer.append((state, action, reward,done, next_state))
        #print("POST",next_state)
        #time.sleep(1)

    def sample_batch(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        #print(batch[3])
        #time.sleep(5)
        states, actions, rewards,dones, next_states = zip(*batch)
        # for elem in states:
        #     print(elem)
    
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(np.array(actions), dtype=torch.float32),
                torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.array(next_states), dtype=torch.float32))
                
    


    def size(self):
        return len(self.buffer)
    
    def burn_in_capacity(self):
        return len(self.buffer) / self.burn_in