import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Value(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims, alpha):
        super(Value, self).__init__()
        self.input_layer = nn.Linear(state_dims, hidden_dims[0])

        self.hidden_layers = nn.ModuleList()
        input_dims = hidden_dims[0] + action_dims
        for i in range(len(hidden_dims)-1):
            hidden_layer = nn.Linear(
                input_dims, hidden_dims[i+1])
            self.hidden_layers.append(hidden_layer)
            input_dims = hidden_dims[i+1]
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        self.activation = F.relu

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state, action):
        x = torch.tensor(state, dtype=torch.float32).to(self.device)
        y = torch.tensor(action, dtype=torch.float32).to(self.device)

        x = self.activation(self.input_layer(x))
        x = torch.cat((x, y), dim=1)
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = self.output_layer(x)
        return x


class Policy(nn.Module):
    def __init__(self, state_dims, action_dims, action_ranges, hidden_dims, alpha):
        super(Policy, self).__init__()
        self.input_layer = nn.Linear(state_dims, hidden_dims[0])
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)
        ])
        self.output_layer = nn.Linear(hidden_dims[-1], action_dims)

        self.activation = F.relu

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # prepare these variables for rescaling
        self.min = torch.tanh(torch.tensor(
            [float('-inf')], dtype=torch.float32)).to(self.device)
        self.max = torch.tanh(torch.tensor(
            [float('inf')], dtype=torch.float32)).to(self.device)
        self.action_min = torch.tensor(
            action_ranges[0], dtype=torch.float32).to(self.device)
        self.action_max = torch.tensor(
            action_ranges[1], dtype=torch.float32).to(self.device)

    def forward(self, state):
        x = torch.tensor(state, dtype=torch.float32).to(self.device)
        x = self.activation(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = self.activation(hidden_layer(x))
        x = torch.tanh(self.output_layer(x))
        return self.rescale(x)

    def rescale(self, x):
        # use the following rescaling function
        # scale from [min, max] to [action_min, action_max]
        # f(x) = ((action_max - action_min) * (x - min)) / (max - min) + action_min
        return ((self.action_max - self.action_min) * (x - self.min)) / (self.max - self.min) + self.action_min
