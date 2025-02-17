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


########################### HUMANOID ########################
# Actor Network
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = torch.tanh(self.fc3(x)) * self.max_action
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc_state = nn.Linear(state_dim, 512)
        self.bn_state = nn.BatchNorm1d(512)
        self.fc1 = nn.Linear(512 + action_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1)
        
    def forward(self, state, action):
        state = self.fc_state(state)
        state = self.bn_state(state)
        state = torch.relu(state)
        
        x = torch.cat([state, action], dim=1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        return x
#####################################################################
# ########################### HOPPER ########################
# # Actor Network
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 256)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.fc2 = nn.Linear(256, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.fc3 = nn.Linear(256, action_dim)
#         self.max_action = max_action
        
#     def forward(self, state):
#         x = self.fc1(state)
#         x = self.bn1(x)
#         x = torch.relu(x)
        
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = torch.relu(x)
        
#         x = torch.tanh(self.fc3(x)) * self.max_action
#         return x

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         self.fc_state = nn.Linear(state_dim, 256)
#         self.bn_state = nn.BatchNorm1d(256)
#         self.fc1 = nn.Linear(256 + action_dim, 256)
#         self.bn1 = nn.BatchNorm1d(256)
#         self.fc2 = nn.Linear(256, 256)
#         self.bn2 = nn.BatchNorm1d(256)
#         self.fc3 = nn.Linear(256, 1)
        
#     def forward(self, state, action):
#         state = self.fc_state(state)
#         state = self.bn_state(state)
#         state = torch.relu(state)
        
#         x = torch.cat([state, action], dim=1)
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = torch.relu(x)
        
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = torch.relu(x)
        
#         x = self.fc3(x)
#         return x
# #####################################################################


# ########################### HALF CHEETA########################
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, action_dim),
#             nn.Tanh()
#         )
#         self.max_action = max_action

#     def forward(self, state):
#         return self.max_action * self.net(state)

# # Definizione della rete Critic
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(state_dim + action_dim, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def forward(self, state, action):
#         return self.net(torch.cat([state, action], dim=1))
# #####################################################################

################CARTPOLE SOLVER##################################

# Actor Network
# class Actor(nn.Module):
#     def __init__(self, state_dim, action_dim, max_action):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, action_dim)
#         self.max_action = max_action
        
#     def forward(self, state):
#         x = torch.relu(self.fc1(state))
#         x = torch.relu(self.fc2(x))
#         x = torch.tanh(self.fc3(x)) * self.max_action
#         return x

# Critic Network
# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(state_dim + action_dim, 256)
#         self.fc2 = nn.Linear(256, 256)
#         self.fc3 = nn.Linear(256, 1)
        
#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
# #######################################################