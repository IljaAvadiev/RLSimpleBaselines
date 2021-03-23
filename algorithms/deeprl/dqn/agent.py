from algorithms.deeprl.dqn.model import Q, DEVICE
from algorithms.deeprl.common.memory import ReplayBuffer
from copy import deepcopy
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


class DQN():
    def __init__(self,
                 env,
                 double=False,
                 duelling=False,
                 activation=F.relu,
                 optimizer=optim.Adam,
                 alpha=0.0001,
                 gamma=0.99,
                 epsilon_start=1,
                 epsilon_end=0.05,
                 epsilon_decay=0.000035,
                 tau=1,
                 max_memory_size=1000000,
                 batch_size=64,
                 max_episodes=1000,
                 warmup=100,
                 replace_steps=100,
                 seed=49,
                 log=False,
                 dir='tmp',
                 name='name'):

        self.env = env
        self.set_seed(seed)
        self.double = double
        self.state_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.n
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau

        self.max_episodes = max_episodes
        self.warmup = warmup
        self.replace_steps = replace_steps
        self.log = log

        self.eval_rewards = []
        self.eval_rewards_mean = []

        self.memory = ReplayBuffer(
            self.state_dims, 1, max_memory_size, batch_size)

        self.q_online = Q(self.state_dims, self.action_dims,
                          duelling, activation, dir, name + '.pt')
        self.q_target = deepcopy(self.q_online)

        for param in self.q_target.parameters():
            param.requires_grad = False

        self.device = DEVICE
        self.optimizer = optimizer(self.q_online.parameters(), alpha)

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.env.seed(seed)
        # TODO cuda seed

    @torch.no_grad()
    def act_greedy(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        state = state.unsqueeze(dim=0)
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

    def update_target_network(self):
        with torch.no_grad():
            for online, target in zip(self.q_online.parameters(), self.q_target.parameters()):
                new_value = self.tau * online.data + \
                    (1 - self.tau) * target.data
                target.data.copy_(new_value)

    def learn(self):
        step = 0
        rewards = []
        rewards_mean = []

        best_eval_reward = float('-inf')
        self.eval_rewards = []
        self.eval_rewards_mean = []

        for episode in range(self.max_episodes):
            state, done = self.env.reset(), False
            reward_sum = 0
            while not done:
                step += 1
                action = self.act_eps_greedy(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.add_memory(state, action, next_state, reward, done)
                state = next_state
                reward_sum += reward
                if step > self.warmup:
                    self.optimize()
                    self.decrease_epsilon()
                if step % self.replace_steps == 0:
                    self.update_target_network()
            rewards.append(reward_sum)
            # evaluation step
            eval_reward = self.evaluate()
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                self.q_online.save()
            self.eval_rewards.append(eval_reward)
            if episode >= 100:
                mean = np.mean(rewards[-100:])
                eval_mean = np.mean(self.eval_rewards[-100:])
                rewards_mean.append(mean)
                self.eval_rewards_mean.append(eval_mean)
                if self.log:
                    print('--------------------------------------------------------')
                    print(f'Episode: {episode}')
                    print(f'Step: {step}')
                    print(f'Evaluation Reward: {eval_reward}')
                    print(f'Best Evaluation Reward: {best_eval_reward}')
                    print(f'Train Mean: {mean}')
                    print(f'Eval Mean: {eval_mean}')
                    print('--------------------------------------------------------')

    def evaluate(self):
        state, done = self.env.reset(), False
        reward_sum = 0
        while not done:
            action = self.act_greedy(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            reward_sum += reward
        return reward_sum
