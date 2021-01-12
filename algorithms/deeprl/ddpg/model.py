import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Q(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims, activation, dir, name):
        super(Q, self).__init__()

        assert len(hidden_dims) == 2
        self.input_layer = nn.Linear(state_dims, hidden_dims[0])
        self.hidden_layer = nn.Linear(
            hidden_dims[0] + action_dims, hidden_dims[1])
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        self.activation = activation
        self.path = os.path.join(dir, name)
        self.to(DEVICE)

    def forward(self, state, action):
        x = state
        y = action
        x = self.activation(self.input_layer(x))
        x = torch.cat((x, y), dim=1)
        x = self.activation(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def save(self):
        torch.save(self.state_dict(), self.path)

    def load(self):
        self.load_state_dict(torch.load(self.path))


class Pi(nn.Module):
    def __init__(self, state_dims, action_dims, action_limit, hidden_dims, activation, dir, name):
        super(Pi, self).__init__()
        assert len(hidden_dims) == 2
        self.input_layer = nn.Linear(state_dims, hidden_dims[0])
        self.hidden_layer = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.output_layer = nn.Linear(hidden_dims[-1], action_dims)

        self.action_limit = action_limit
        self.activation = activation
        self.path = os.path.join(dir, name)
        self.to(DEVICE)

    def forward(self, state):
        x = state
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer(x))
        x = torch.tanh(self.output_layer(x))
        return x * self.action_limit

    def save(self):
        torch.save(self.state_dict(), self.path)

    def load(self):
        self.load_state_dict(torch.load(self.path))
