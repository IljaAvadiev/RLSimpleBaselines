import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


