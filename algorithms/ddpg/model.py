import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy


class Value(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims, alpha):
        super(Value, self).__init__()
        self.input_layer = nn.Linear(state_dims, hidden_dims[0])
        self.hidden_layers = nn.ModuleList()
        for hidden_dim in range(len(hidden_dims)-1):
            if hidden_dim == 0:
                hidden_layer = nn.Linear(
                    hidden_dims[0] + action_dims, hidden_dims[1])
            else:
                hidden_layer = nn.Linear(
                    hidden_dims[hidden_dim], hidden_dims[hidden_dim+1])
            self.hidden_layers.append(hidden_layer)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        self.activation = F.relu

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state, action):
        x = torch.tensor(state, dtype=torch.float32).to(self.device)
        y = torch.tensor(action, dtype=torch.float32).to(self.device)

        x = self.activation(self.input_layer(x))
        x = self.activation(torch.cat((x, y)))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x


class Policy(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims, alpha):
        self.input_layer = nn.Linear(state_dims, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)
        ])
        self.output_layer = nn.Linear(hidden_dims[-1], action_dims)

        self.activation = F.relu

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        x = torch.tensor(state, dtype=float32).to(self.device)
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = F.tanh(self.output_layer(x))
        return x
