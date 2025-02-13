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

#### FRANCESCA #############
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import random
import os
import pathlib

# # ---- ACTOR NETWORK ----
# class Actor(nn.Module):
#     def __init__(self, action_dim, max_action, state_dim,name= "gg",learning_rate=1e-2, fc1_dims=256, fc2_dims=256,ckpt_dir= 'ckpt'):
#         super(Actor, self).__init__()
#         self.max_action = max_action
#         self.learning_rate = learning_rate
#         self.state_dim = state_dim
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.action_dim = action_dim
#         self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg')

#         self.fc1 = nn.Linear(self.state_dim, self.fc1_dims)
#         self.bn1 = nn.LayerNorm(self.fc1_dims)

#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
#         self.bn2 = nn.LayerNorm(self.fc2_dims)
#         self.fc3 = nn.Linear(self.fc2_dims, self.action_dim)
#         self.actor_optimizer = None
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
#         x = F.relu(self.bn1(self.fc1(state)))
#         x = F.relu(self.bn2(self.fc2(x)))
#         x = torch.tanh(self.fc3(x))*self.max_action # Azioni in [-1, 1]
#         return x
#     # def save_checkpoint(self):
#     #     torch.save(self.state_dict(), self.checkpoint_file)

#     # def load_checkpoint(self):
#     #     self.load_state_dict(torch.load(self.checkpoint_file))
# # ---- CRITIC NETWORK ----
# class Critic(nn.Module):
#     def __init__(self, action_dim, state_dim,learning_rate=1e-2, fc1_dims=256, fc2_dims=256,name = "gg",ckpt_dir= 'ckpt'):
#         super(Critic, self).__init__()
#         self.learning_rate = learning_rate
#         self.state_dim = state_dim
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.action_dim = action_dim
#         self.name = name

#         self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg')
        
#         self.fc1 = nn.Linear(self.state_dim, self.fc1_dims)
#         self.bn1 = nn.LayerNorm(self.fc1_dims)
#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
#         self.bn2 = nn.LayerNorm(self.fc2_dims)
#         self.action_value = nn.Linear(self.action_dim, self.fc2_dims)
#         self.q = nn.Linear(self.fc2_dims, 1)
#         self.critic_optimizer = None
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

#     def save_checkpoint(self):
#         torch.save(self.state_dict(), self.checkpoint_file)

#     def load_checkpoint(self):
#         self.load_state_dict(torch.load(self.checkpoint_file))
# ###############################################################
#Actor Network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x

# Critic Network
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
        
#         # Convolutional layers
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=2)  # (84x96 -> 21x24)
#         self.bn1 = nn.BatchNorm2d(32)  # BatchNorm for conv1

#         self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # (21x24 -> 10x11)
#         self.bn2 = nn.BatchNorm2d(64)  # BatchNorm for conv2

#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)  # (10x11 -> 10x11)
#         self.bn3 = nn.BatchNorm2d(64)  # BatchNorm for conv3

#         # Fully connected layer
#         self.fc = nn.Linear(64 * 10 * 12, 512)  # Adjusted input size
#         self.bn_fc = nn.BatchNorm1d(512)  # BatchNorm for fully connected layer

#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))  # Conv1 -> BatchNorm -> ReLU
#         x = F.relu(self.bn2(self.conv2(x)))  # Conv2 -> BatchNorm -> ReLU
#         x = F.relu(self.bn3(self.conv3(x)))  # Conv3 -> BatchNorm -> ReLU

#         x = x.view(x.size(0), -1)  # Flatten
#         x = F.relu(self.bn_fc(self.fc(x)))  # Fully connected -> BatchNorm -> ReLU
#         return x


# class Actor(nn.Module):
#     def __init__(self, action_dim):
#         super(Actor, self).__init__()
#         self.cnn = SimpleCNN()
#         self.fc1 = nn.Linear(512, 256)
#         self.fc2 = nn.Linear(256, action_dim)

#     def forward(self, state):
#         # state = state.view(-1, 3, 96, 96)  # Reshape corretto

#         x = self.cnn(state)
#         x = F.relu(self.fc1(x))
#         #action = torch.tanh(self.fc2(x))  # Output in [-1,1]
#         steering = torch.tanh(self.fc2(x)[:, 0])  # [-1, 1]
#         acceleration = torch.sigmoid(self.fc2(x)[:, 1])  # [0, 1]
#         brake = torch.sigmoid(self.fc2(x)[:, 2])  # [0, 1]
#         return torch.stack([steering, acceleration, brake], dim=-1)
       


# class Critic(nn.Module):
#     def __init__(self, action_dim):
#         super(Critic, self).__init__()
#         self.cnn = SimpleCNN()
#         self.fc1 = nn.Linear(512 + action_dim, 256)  # Concat feature + action
#         self.fc2 = nn.Linear(256, 1)

#     def forward(self, state, action):
#         # state = state.view(-1, 3, 96, 96)  # Reshape corretto
#         x = self.cnn(state)
#         # print("ACTION",action.shape)
#         x = torch.cat([x, action], dim=-1)  # Concat features & action
#         x = F.relu(self.fc1(x))
#         value = self.fc2(x)
#         return value


# # ---- ACTOR NETWORK ----
# class Actor(nn.Module):
#     def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, action_dim, name, ckpt_dir= 'ckpt'):
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


    #def save_checkpoint(self):
       # torch.save(self.state_dict(), self.checkpoint_file)

    #def load_checkpoint(self):
    #    self.load_state_dict(torch.load(self.checkpoint_file))
