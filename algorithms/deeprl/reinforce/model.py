import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Pi(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims, activation, dir, name):
        super(Pi, self).__init__()
        self.input_layer = nn.Linear(state_dims, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims) - 1)])
        self.output_layer = nn.Linear(hidden_dims[-1], action_dims)

        self.activation = activation
        self.path = os.path.join(dir, name)
        self.to(DEVICE)

    def forward(self, state):
        x = state
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        distribution = Categorical(logits=x)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action, log_prob

    def save(self):
        torch.save(self.state_dict(), self.path)

    def load(self):
        self.load_state_dict(torch.load(self.path))
