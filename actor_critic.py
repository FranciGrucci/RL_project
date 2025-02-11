import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import random
import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10): 
        super(SimpleCNN, self).__init__()
        # Primo blocco convoluzionale
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)  # output: 16 x 84 x 96
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 16 x 42 x 48

        # Secondo blocco convoluzionale
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)  # output: 32 x 42 x 48
        # Dopo MaxPooling: output: 32 x 21 x 24

        # Terzo blocco convoluzionale
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)  # output: 64 x 21 x 24
        # Dopo MaxPooling: output: 64 x 10 x 12

        # Fully connected layer
        self.fc1 = nn.Linear(in_features=64 * 10 * 12, out_features=128)  # Primo livello fully connected
        self.fc2 = nn.Linear(in_features=128, out_features=512)  # Output finale

    def forward(self, x):
        # Passaggio nei layer convoluzionali con ReLU e MaxPooling
        x = self.pool(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv2(x)))  # Conv2 -> ReLU -> MaxPool
        x = self.pool(F.relu(self.conv3(x)))  # Conv3 -> ReLU -> MaxPool
        
        # Flatten per il passaggio al fully connected
        x = x.view(-1, 64 * 10 * 12)  # Flatten (batch_size, features)
        
        # Passaggio nei layer fully connected
        x = F.relu(self.fc1(x))  # Primo fully connected con ReLU
        x = self.fc2(x)  # Output finale
        return x

class Actor(nn.Module):
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.cnn = SimpleCNN()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, action_dim)
    
    def forward(self, state):
        #state = state.view(-1, 3, 96, 96)  # Reshape corretto
        
        x = self.cnn(state)
        x = F.relu(self.fc1(x))
        action = torch.tanh(self.fc2(x))  # Output in [-1,1]
        return action

class Critic(nn.Module):
    def __init__(self, action_dim):
        super(Critic, self).__init__()
        self.cnn = SimpleCNN()
        self.fc1 = nn.Linear(512 + action_dim, 256)  # Concat feature + action
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, state, action):
        #state = state.view(-1, 3, 96, 96)  # Reshape corretto
        x = self.cnn(state)
        #print("ACTION",action.shape)
        x = torch.cat([x, action], dim=-1)  # Concat features & action
        x = F.relu(self.fc1(x))
        value = self.fc2(x)
        return value


# # ---- ACTOR NETWORK ----
# class Actor(nn.Module):
#     def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions, name, ckpt_dir= 'ckpt'):
#         super(Actor, self).__init__()
#         self.learning_rate = learning_rate
#         self.input_dims = input_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.n_actions = n_actions
#         self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg')

#         self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
#         self.bn1 = nn.LayerNorm(self.fc1_dims)

#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
#         self.bn2 = nn.LayerNorm(self.fc2_dims)
#         self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
#         self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
#         #self.device = torch.device("cpu")
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         self.to(self.device)

#         if not os.path.exists(ckpt_dir):
#             pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

#     #inizializzazione dei pesi:

#     def init_weights(self):
#         f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
#         torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
#         torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
#         f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
#         torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
#         torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        
#         f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
#         torch.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
#         torch.nn.init.uniform_(self.fc3.bias.data, -f3, f3)

#     def forward(self, state):
#         state = state.to(self.device)
#         #print(state)
#         x = F.relu(self.bn1(self.fc1(state)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = torch.tanh(self.fc3(x)) # Azioni in [-1, 1]
#         return x
#     # def save_checkpoint(self):
#     #     torch.save(self.state_dict(), self.checkpoint_file)

#     # def load_checkpoint(self):
#     #     self.load_state_dict(torch.load(self.checkpoint_file))

# # ---- CRITIC NETWORK ----
# class Critic(nn.Module):
#     def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions, name, ckpt_dir= 'ckpt'):
#         super(Critic, self).__init__()
#         self.learning_rate = learning_rate
#         self.input_dims = input_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.n_actions = n_actions
#         self.name = name

#         self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg')
        
#         self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
#         self.bn1 = nn.LayerNorm(self.fc1_dims)
#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
#         self.bn2 = nn.LayerNorm(self.fc2_dims)
#         self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
#         self.q = nn.Linear(self.fc2_dims, 1)
#         self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
#         #self.device = torch.device('cpu')
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
#         self.to(self.device)


#     #inizializzazione dei pesi:

#     # Inizializzazione dei pesi
#     def init_weights(self):
#          f1 = 1 /np.sqrt(self.fc1.weight.data.size()[0])
#          torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
#          torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

         
#          f2 = 1 /np.sqrt(self.fc2.weight.data.size()[0])
#          torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
#          torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        
#          f_action = 1 /np.sqrt(self.action_value.weight.data.size()[0])
#          torch.nn.init.uniform_(self.action_value.weight.data, -f_action, f_action)
#          torch.nn.init.uniform_(self.action_value.bias.data, -f_action, f_action)
#          f_q = 1 /np.sqrt(self.q.weight.data.size()[0])
#          torch.nn.init.uniform_(self.q.weight.data, -f_q, f_q)
#          torch.nn.init.uniform_(self.q.bias.data, -f_q, f_q)
         
        

#     # def forward(self, state, action):
#     #     state_value = F.relu(self.bn1(self.fc1(state)))
#     #     state_value = F.relu(self.bn2(self.fc2(state_value)))
#     #     action_value = F.relu(self.action_value(action))
#     #     state_action_value = F.relu(torch.add(state_value, action_value))
#     #     return self.q(state_action_value)
#     def forward(self, state, action):
#         state = state.to(self.device)
#         state_value = self.fc1(state)
#         state_value = self.bn1(state_value)
#         state_value = F.relu(state_value)

#         state_value = self.fc2(state_value)
#         state_value = self.bn2(state_value)
        
#         action_value = F.relu(self.action_value(action))
#         state_action_value = F.relu(torch.add(state_value, action_value))
#         return self.q(state_action_value)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
