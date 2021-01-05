import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dim1, hidden_dim2, lr, name, dir='tmp'):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dims, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, action_dims)

        self.path = os.path.join(dir, name)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = state
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.tanh(x)

        return x

    def save(self):
        torch.save(self.state_dict(), self.path)

    def load(self):
        self.load_state_dict(torch.load(self.path))


class Critic(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dim1, hidden_dim2, lr, name, dir='tmp'):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dims + action_dims, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, 1)

        self.path = os.path.join(dir, name)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

    def save(self):
        torch.save(self.state_dict(), self.path)

    def load(self):
        self.load_state_dict(torch.load(self.path))
