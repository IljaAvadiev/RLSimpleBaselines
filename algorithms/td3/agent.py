import torch
import numpy as np
from model import Actor, Critic
from replay_buffer import ReplayBuffer
from copy import deepcopy


class Agent():
    def __init__(self, state_dims, action_dims, hidden_dim1, hidden_dim2, lr_actor, lr_critic, gamma, tau, mem_size, batch_size, max_action, min_action, warmup):
        self.memory = ReplayBuffer(
            state_dims=state_dims, action_dims=action_dims, max_memsize=mem_size, batch_size=batch_size)
        self.actor = Actor(
            state_dims, action_dims, hidden_dim1, hidden_dim2, lr=lr_actor, name='actor_td3')
        self.actor_target = deepcopy(self.actor)

        self.critic_1 = Critic(state_dims, action_dims,
                               hidden_dim1, hidden_dim2, lr_critic, 'critic_1_td3')
        self.critic_2 = Critic(state_dims, action_dims,
                               hidden_dim1, hidden_dim2, lr_critic, 'critic_2_td3')
        self.critic_1_target = deepcopy(self.critic_1)
        self.critic_2_target = deepcopy(self.critic_2)
        self.min = min_action
        self.max = max_action
        self.gamma = gamma
        self.tau = tau
        self.action_dims = action_dims
        self.count = 0
        self.warmup = warmup
        self.device = self.actor.device

    def choose_action(self, state):
        self.count += 1
        with torch.no_grad():
            mu = 0
            sigma = 0.1
            noise = torch.tensor(np.random.normal(
                mu, sigma, size=self.action_dims), dtype=torch.float32).to(self.device)
            if self.count < self.warmup:
                # take purely random action
                action = noise
            else:
                state = torch.tensor(
                    state, dtype=torch.float32).to(self.device)
                action = self.actor(state)
                action = action + noise

            return torch.clamp(action, self.min, self.max).cpu().detach().numpy()

    def greedy_action(self, state):
        with torch.no_grad():
            state = torch.tensor(
                state, dtype=torch.float32).to(self.device)
            action = self.actor(state)
            return action.cpu().detach().numpy()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.add_memory(state, action, reward, next_state, done)

    def learn(self):
        if self.count < self.warmup:
            return
        states, actions, rewards, next_states, terminals = self.memory.sample()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float32).to(self.device)
        terminals = torch.tensor(
            terminals, dtype=torch.float32).to(self.device)

        next_actions = self.actor_target(next_states).detach()
        noise = torch.clamp(torch.normal(
            0, 0.2, size=next_actions.shape), -0.5, 0.5)
        next_actions += noise

        next_actions = torch.clamp(next_actions, self.min, self.max)

        target_1 = self.critic_1_target(next_states, next_actions)
        target_2 = self.critic_2_target(next_states, next_actions)

        target = rewards + \
            self.gamma * torch.min(target_1, target_2) * \
            torch.logical_not(terminals)

        online_1 = self.critic_1(states, actions)
        online_2 = self.critic_2(states, actions)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        loss_fn = torch.nn.MSELoss()
        loss_1 = loss_fn(target, online_1)
        loss_2 = loss_fn(target, online_2)
        critic_loss = loss_1 + loss_2

        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        if self.count % 2 == 0:
            self.actor.optimizer.zero_grad()
            actor_loss = -self.critic_1(states, self.actor(states)).mean()
            actor_loss.backward()
            self.actor.optimizer.step()

            self.update_target_weights()

    def update_target_weights(self):
        actor_params = self.actor.named_parameters()
        actor_target_params = self.actor_target.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        critic_1_target_params = self.critic_1_target.named_parameters()
        critic_2_target_params = self.critic_2_target.named_parameters()

        actor_state_dict = dict(actor_params)
        actor_target_state_dict = dict(actor_target_params)
        critic_1_state_dict = dict(critic_1_params)
        critic_2_state_dict = dict(critic_2_params)
        critic_1_target_state_dict = dict(critic_1_target_params)
        critic_2_target_state_dict = dict(critic_2_target_params)

        for name in critic_1_state_dict:
            critic_1_state_dict[name] = self.tau*critic_1_state_dict[name].clone() + \
                (1-self.tau)*critic_1_target_state_dict[name].clone()

        for name in critic_2_state_dict:
            critic_2_state_dict[name] = self.tau*critic_2_state_dict[name].clone() + \
                (1-self.tau)*critic_2_target_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = self.tau*actor_state_dict[name].clone() + \
                (1-self.tau)*actor_target_state_dict[name].clone()

        self.critic_1_target.load_state_dict(critic_1_state_dict)
        self.critic_2_target.load_state_dict(critic_2_state_dict)
        self.actor_target.load_state_dict(actor_state_dict)

    def save(self):
        self.actor.save()
        self.critic_1.save()
        self.critic_2.save()

    def load(self):
        self.actor.load()
        self.critic_1.save()
        self.critic_2.save()
