from algorithms.deeprl.dqn.model import Q, DEVICE
from algorithms.deeprl.common.memory import ReplayBuffer
from itertools import count
from copy import deepcopy
import torch
import numpy as np
from gym import wrappers


class DQN():
    def __init__(self, env, double, state_dims, action_dims, hidden_dims, activation, optimizer,
                 alpha, gamma, epsilon_start, epsilon_end, epsilon_decay,
                 max_memory_size, batch_size, dir, name):

        self.env = env
        self.double = double
        self.action_dims = action_dims
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.memory = ReplayBuffer(
            state_dims, 1, max_memory_size, batch_size)

        self.q_online = Q(state_dims, action_dims,
                          hidden_dims, activation, dir, name + '.pt')
        self.q_target = deepcopy(self.q_online)
        self.device = DEVICE
        self.optimizer = optimizer(self.q_online.parameters(), alpha)

    @torch.no_grad()
    def act_greedy(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        actions = self.q_online(state).cpu().detach().numpy()
        greedy_action = np.argmax(actions)
        return greedy_action

    @torch.no_grad()
    def act_eps_greedy(self, state):
        if np.random.rand() > self.epsilon:
            action = self.act_greedy(state)
        else:
            action = np.random.choice(self.action_dims)
        return action

    def decrease_epsilon(self):
        self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end

    def sample_batch(self):
        states, actions, next_states, rewards, terminals = self.memory.sample_batch()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        terminals = torch.tensor(
            terminals, dtype=torch.float32).to(self.device)

        return states, actions, next_states, rewards, terminals

    def optimize(self):
        self.optimizer.zero_grad()
        states, actions, next_states, rewards, terminals = self.sample_batch()

        with torch.no_grad():
            if self.double:
                indices = torch.max(self.q_online(
                    next_states).detach(), dim=1, keepdim=True)[1]
            else:
                indices = torch.max(self.q_target(
                    next_states).detach(), dim=1, keepdim=True)[1]

            target = rewards + self.gamma * \
                self.q_target(next_states).gather(
                    dim=1, index=indices).detach() * torch.logical_not(terminals)

        online = self.q_online(states).gather(dim=1, index=actions)

        error = target-online
        loss = error.pow(2).mul(0.5).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_online.parameters(), 1.0)
        self.optimizer.step()

    def replace_target_network(self):
        self.q_target = deepcopy(self.q_online)

    def learn(self, max_episodes, warmup, replace_steps, average_len, target_reward, log=False):
        step = 0
        rewards = []
        rewards_mean = []

        eval_rewards = []
        eval_rewards_mean = []

        for episode in range(max_episodes):
            state, done = self.env.reset(), False
            reward_sum = 0
            while not done:
                step += 1
                action = self.act_eps_greedy(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.add_memory(state, action, next_state, reward, done)
                state = next_state
                reward_sum += reward
                if step > warmup:
                    self.optimize()
                    self.decrease_epsilon()
                if step % replace_steps == 0:
                    self.replace_target_network()
            rewards.append(reward_sum)
            eval_rewards.append(self.evaluate())
            if episode >= average_len:
                mean = np.mean(rewards[-average_len:])
                eval_mean = np.mean(eval_rewards[-average_len:])
                rewards_mean.append(mean)
                eval_rewards_mean.append(mean)
                if log:
                    print('--------------------------------------------------------')
                    print(f'Episode: {episode}')
                    print(f'Step: {step}')
                    print(f'Train Mean: {mean}')
                    print(f'Eval Mean: {eval_mean}')
                    print('--------------------------------------------------------')
                    if eval_mean >= target_reward:
                        print('GOAL ACHIEVED!')
                        self.q_online.save()
                        break

    def evaluate(self):
        state, done = self.env.reset(), False
        reward_sum = 0
        while not done:
            action = self.act_greedy(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            reward_sum += reward
        return reward_sum
