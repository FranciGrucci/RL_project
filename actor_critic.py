import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import random
import os
import pathlib

# ---- ACTOR NETWORK ----
class Actor(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions, name, ckpt_dir= 'ckpt'):
        super(Actor, self).__init__()
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims) 
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)

        if not os.path.exists(ckpt_dir):
            pathlib.Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

    #inizializzazione dei pesi:

    def init_weights(self):
        f1 = 1 / np.sqrt(self.fc1.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        
        f2 = 1 / np.sqrt(self.fc2.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        
        f3 = 1 / np.sqrt(self.fc3.weight.data.size()[0])
        torch.nn.init.uniform_(self.fc3.weight.data, -f3, f3)
        torch.nn.init.uniform_(self.fc3.bias.data, -f3, f3)

    def forward(self, state):
        state = state.to(self.device)
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x)) # Azioni in [-1, 1]
        return x
    # def save_checkpoint(self):
    #     torch.save(self.state_dict(), self.checkpoint_file)

    # def load_checkpoint(self):
    #     self.load_state_dict(torch.load(self.checkpoint_file))

# ---- CRITIC NETWORK ----
class Critic(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions, name, ckpt_dir= 'ckpt'):
        super(Critic, self).__init__()
        self.learning_rate = learning_rate
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name

        self.checkpoint_file = os.path.join(ckpt_dir, name+'_ddpg')
        
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #self.device = torch.device('cpu')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.to(self.device)


    #inizializzazione dei pesi:

    # Inizializzazione dei pesi
    def init_weights(self):
         f1 = 1 /np.sqrt(self.fc1.weight.data.size()[0])
         torch.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
         torch.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

         
         f2 = 1 /np.sqrt(self.fc2.weight.data.size()[0])
         torch.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
         torch.nn.init.uniform_(self.fc2.bias.data, -f2, f2)
        
         f_action = 1 /np.sqrt(self.action_value.weight.data.size()[0])
         torch.nn.init.uniform_(self.action_value.weight.data, -f_action, f_action)
         torch.nn.init.uniform_(self.action_value.bias.data, -f_action, f_action)
         f_q = 1 /np.sqrt(self.q.weight.data.size()[0])
         torch.nn.init.uniform_(self.q.weight.data, -f_q, f_q)
         torch.nn.init.uniform_(self.q.bias.data, -f_q, f_q)
         
        

    # def forward(self, state, action):
    #     state_value = F.relu(self.bn1(self.fc1(state)))
    #     state_value = F.relu(self.bn2(self.fc2(state_value)))
    #     action_value = F.relu(self.action_value(action))
    #     state_action_value = F.relu(torch.add(state_value, action_value))
    #     return self.q(state_action_value)
    def forward(self, state, action):
        state = state.to(self.device)
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        
        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(torch.add(state_value, action_value))
        return self.q(state_action_value)

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
