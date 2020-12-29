import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np


# this part is inspired by OpenAi spinning up
# gaussian policy
class Actor(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims, alpha, log_sigma_min, log_sigma_max):
        super(Actor, self).__init__()
        self.input_layer = nn.Linear(state_dims, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i+1])
             for i in range(len(hidden_dims)-1)]
        )
        self.mu_layer = nn.Linear(hidden_dims[-1], 1)
        self.log_sigma_layer = nn.Linear(hidden_dims[-1], 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def forward(self, state, greedy=False):
        x = F.relu(self.input_layer(state))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        mu = self.mu_layer(x)
        log_sigma = self.log_sigma_layer(x)
        log_sigma = torch.clamp(
            log_sigma, self.log_sigma_min, self.log_sigma_max)
        sigma = torch.exp(log_sigma)

        distribution = Normal(mu, sigma)

        if greedy:
            # greedy action is just the mean
            action = mu
        else:
            action = distribution.rsample()

        log_prob = distribution.log_prob(action).sum(axis=-1)
        log_prob -= (2*(np.log(2) - action -
                        F.softplus(-2*action))).sum(axis=1)

        action = torch.tanh(action)
        log_prob = log_prob.unsqueeze(1)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dims, action_dims, hidden_dims, alpha):
        super(Critic, self).__init__()
        self.input_layer = nn.Linear(state_dims + action_dims, hidden_dims[0])
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims[i], hidden_dims[i+1]) for i in range(len(hidden_dims)-1)])
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x
