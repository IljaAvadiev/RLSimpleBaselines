import numpy as np
import torch
from algorithms.deeprl.ddpg.model import Pi, Q, DEVICE
from algorithms.deeprl.common.memory import ReplayBuffer
from copy import deepcopy


# implementation from OpenAi
class OrnsteinUhlenbeckActionNoise():
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * \
            np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(
            self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


class DDPG():
    def __init__(self, env, state_dims, action_dims, action_limit, hidden_dims, activation, optimizer,
                 pi_alpha, q_alpha, gamma, tau, max_memory_size, batch_size, dir, name):

        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.action_limit = action_limit

        self.online_policy = Pi(
            state_dims, action_dims, action_limit, hidden_dims, activation, dir, name)
        self.online_value = Q(state_dims, action_dims,
                              hidden_dims, activation, dir, name)

        self.target_policy = deepcopy(self.online_policy)
        self.target_value = deepcopy(self.online_value)

        self.device = DEVICE
        self.policy_optimizer = optimizer(
            self.online_policy.parameters(), pi_alpha)
        self.value_optimizer = optimizer(
            self.online_value.parameters(), q_alpha, weight_decay=1e-2)

        # no grad calculations are required for target networks due to polyak averaging
        for params in self.target_policy.parameters():
            params.requires_grad = False
        for params in self.target_value.parameters():
            params.requires_grad = False

        self.noise = OrnsteinUhlenbeckActionNoise(
            mu=np.zeros(action_dims))
        self.memory = ReplayBuffer(state_dims, action_dims,
                                   max_memory_size=max_memory_size, batch_size=batch_size)

    def reset(self):
        self.noise.reset()

    @torch.no_grad()
    def act_greedy(self, state):
        self.online_policy.eval()
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = self.online_policy(state)
        self.online_policy.train()
        action = action.cpu().detach().numpy()
        return action

    @torch.no_grad()
    def act(self, state):
        self.online_policy.eval()
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        noise = torch.tensor(self.noise(), dtype=torch.float32).to(self.device)
        action = self.online_policy(state) + noise
        self.online_policy.train()
        action = action.cpu().detach().numpy()
        return action

    def sample_batch(self):
        states, actions, next_states, rewards, terminals = self.memory.sample_batch()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(
            next_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        terminals = torch.tensor(
            terminals, dtype=torch.float32).to(self.device)

        return states, actions, next_states, rewards, terminals

    def optimize(self):
        states, actions, next_states, rewards, terminals = self.sample_batch()

        with torch.no_grad():
            next_actions = self.target_policy(next_states)
            target = rewards + \
                self.target_value(next_states, next_actions) * \
                torch.logical_not(terminals)
        online = self.online_value(states, actions)

        self.value_optimizer.zero_grad()
        value_loss = (online - target.detach()).pow(2).mul(0.5).mean()
        value_loss.backward()
        self.value_optimizer.step()

        self.policy_optimizer.zero_grad()
        online_actions = self.online_policy(states)
        policy_loss = - (self.online_value(states, online_actions).mean())
        policy_loss.backward()
        self.policy_optimizer.step()

        # polyak averaging
        with torch.no_grad():
            # update q-value target network
            for online, target in zip(self.online_value.parameters(), self.target_value.parameters()):
                target.data.mul_(1-self.tau)
                target.data.add_(self.tau * online.data)
            # update policy target network
            for online, target in zip(self.online_policy.parameters(), self.target_policy.parameters()):
                target.data.mul_(1-self.tau)
                target.data.add_(self.tau * online.data)

    def learn(self, max_episodes, warmup, target_reward, log=False):
        step = 0
        rewards = []
        rewards_mean = []

        eval_rewards = []
        eval_rewards_mean = []

        for episode in range(max_episodes):
            state, done = self.env.reset(), False
            reward_sum = 0
            self.reset()
            while not done:
                step += 1
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.add_memory(state, action, next_state, reward, done)
                state = next_state
                reward_sum += reward
                if step > warmup:
                    self.optimize()

            rewards.append(reward_sum)
            eval_rewards.append(self.evaluate())
            if episode >= 100:
                mean = np.mean(rewards[-100:])
                eval_mean = np.mean(eval_rewards[-100:])
                rewards_mean.append(mean)
                eval_rewards_mean.append(eval_mean)
                if log:
                    print('--------------------------------------------------------')
                    print(f'Episode: {episode}')
                    print(f'Step: {step}')
                    print(f'Score: {reward_sum}')
                    print(f'Train Mean: {mean}')
                    print(f'Eval Score: {eval_rewards[-1]}')
                    print(f'Eval Mean 100: {eval_mean}')
                    print('--------------------------------------------------------')
                    if eval_mean >= target_reward:
                        print('GOAL ACHIEVED!')
                        return eval_rewards_mean
            else:
                print('--------------------------------------------------------')
                print(f'Episode: {episode}')

        return eval_rewards_mean

    def evaluate(self):
        state, done = self.env.reset(), False
        reward_sum = 0
        while not done:
            action = self.act_greedy(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            reward_sum += reward
        return reward_sum
