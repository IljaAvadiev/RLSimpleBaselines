import numpy as np
import torch
from model import Policy, Value
from copy import deepcopy
from noise import OrnsteinUhlenbeckActionNoise
from replay_buffer import ReplayBuffer


class Agent():
    def __init__(self, state_dims, action_dims, action_ranges, hidden_dims, alpha, gamma, tau, max_memsize, batch_size):
        self.gamma = gamma
        self.tau = tau

        self.online_policy = Policy(
            state_dims, action_dims, action_ranges, hidden_dims, alpha)
        self.online_value = Value(state_dims, action_dims, hidden_dims, alpha)

        self.target_policy = deepcopy(self.online_policy)
        self.target_value = deepcopy(self.online_value)

        # no grad calculations are required for target networks due to polyak averaging
        for params in self.target_policy.parameters():
            params.requires_grad = False
        for params in self.target_value.parameters():
            params.requires_grad = False

        self.noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(action_dims))
        self.memory = ReplayBuffer(
            max_memsize=max_memsize, batch_size=batch_size)

    def reset(self):
        self.noise.reset()

    def add_memory(self, state, action, reward, next_state, done):
        self.memory.add_memory(state, action, reward, next_state, done)

    def optimize(self):
        device = self.online_policy.device
        states, actions, rewards, next_states, terminals = self.memory.sample()
        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        next_states = np.stack(next_states)
        terminals = np.stack(terminals)

        next_actions = self.target_policy(next_states)
        target = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1) + \
            self.target_value(next_states, next_actions) * \
            torch.logical_not(torch.tensor(
                terminals, dtype=torch.bool)).unsqueeze(1)
        online = self.online_value(states, actions)

        self.online_value.optimizer.zero_grad()
        value_loss = self.online_value.loss(online, target)
        value_loss.backward()
        self.online_value.optimizer.step()

        self.online_policy.optimizer.zero_grad()

        online_actions = self.online_policy(states)
        policy_loss = - (self.online_value(states, online_actions).mean())
        policy_loss.backward()
        self.online_policy.optimizer.step()

        # polyak averaging
        with torch.no_grad():
            # update q-value target network
            for online, target in zip(self.online_value.parameters(), self.target_value.parameters()):
                target.data.mul_(self.tau)
                target.data.add_((1-self.tau) * online.data)
            # update policy target network
            for online, target in zip(self.online_policy.parameters(), self.target_policy.parameters()):
                target.data.mul_(self.tau)
                target.data.add_((1-self.tau) * online.data)

    def select_action(self, state):
        return self.online_policy.act(state) + self.noise()
