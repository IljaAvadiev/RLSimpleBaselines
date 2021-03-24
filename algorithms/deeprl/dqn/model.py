import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CnnQ(nn.Module):
    def __init__(self, action_dims, activation, dir, name):
        super(CnnQ, self).__init__()
        self.activation = activation

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        with torch.no_grad():
            dummy = torch.zeros(size=(1, 4, 84, 84))
            result = self.convolve(dummy).cpu().detach().numpy()
            self.linear_input_dim = np.product(result.shape)

        self.linear = nn.Linear(self.linear_input_dim, 512)
        self.output = nn.Linear(512, action_dims)

        
        self.path = os.path.join(dir, name)
        self.to(DEVICE)

    def convolve(self, state):
        x = state
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        return x

    def forward(self, state):
        x = state
        x = self.convolve(x)
        x = x.view(-1, self.linear_input_dim)

        x = self.activation(self.linear(x))
        x = self.output(x)

        return x

    def save(self):
        torch.save(self.state_dict(), self.path)

    def load(self):
        self.load_state_dict(torch.load(self.path))


class Q(nn.Module):
    def __init__(self, state_dims, action_dims, duelling, activation, dir, name):
        super(Q, self).__init__()
        hidden_dim_1 = 512
        hidden_dim_2 = 256
        hidden_dim_3 = 64
        self.duelling = duelling
        self.input_layer = nn.Linear(state_dims, hidden_dim_1)
        self.hidden_layer_1 = nn.Linear(hidden_dim_1, hidden_dim_2)

        if self.duelling:
            self.v_input = nn.Linear(hidden_dim_2, hidden_dim_3)
            self.v = nn.Linear(hidden_dim_3, 1)

            self.a_input = nn.Linear(hidden_dim_2, hidden_dim_3)
            self.a = nn.Linear(hidden_dim_3, action_dims)

        else:
            self.hidden_layer_2 = nn.Linear(hidden_dim_2, hidden_dim_3)
            self.output_layer = nn.Linear(hidden_dim_3, action_dims)

        self.activation = activation
        self.path = os.path.join(dir, name)
        self.to(DEVICE)

    def forward(self, state):
        x = state
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer_1(x))

        if self.duelling:
            v = self.activation(self.v_input(x))
            v = self.v(v)

            a = self.activation(self.a_input(x))
            a = self.a(a)

            x = v + a - a.mean(dim=1, keepdim=True)
        else:
            x = self.activation(self.hidden_layer_2(x))
            x = self.output_layer(x)
        return x

    def save(self):
        torch.save(self.state_dict(), self.path)

    def load(self):
        self.load_state_dict(torch.load(self.path))

