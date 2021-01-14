from algorithms.deeprl.vgp.model import Pi, V, DEVICE
import torch
import numpy as np


class A2C():
    def __init__(self, env, state_dims, action_dims, hidden_dims, activation, optimizer,
                 pi_alpha, v_alpha, beta, gamma, tau, dir, name):
        self.env = env
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.pi = Pi(state_dims, action_dims,
                     hidden_dims, activation, dir, name)
        self.v = V(state_dims, hidden_dims, activation, dir, name)
        self.device = DEVICE
        self.policy_optimzer = optimizer(self.pi.parameters(), pi_alpha)
        self.value_optimzer = optimizer(self.v.parameters(), v_alpha)

    def reset(self):
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.entropies = []

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action, log_prob, entropy = self.pi(state)
        value = self.v(state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.entropies.append(entropy)
        return action.cpu().detach().item()

    def next_value(self, state, done):
        if not done:
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            value = self.v(state)
        else:
            value = torch.tensor(0, dtype=torch.float32).to(self.device)
        self.values.append(value)

    def optimize(self):
        trajectory_len = len(self.rewards)

        factors = torch.tensor(
            [(self.gamma*self.tau)**exp for exp in range(trajectory_len)]).view(-1, 1).to(self.device)

        residuals = torch.vstack([self.rewards[i] + self.gamma * self.values[i+1] -
                                  self.values[i] for i in range(trajectory_len)])

        gaes = torch.vstack([torch.sum(residuals[i:] * factors[:trajectory_len-i])
                             for i in range(trajectory_len)])

        log_probs = torch.vstack(self.log_probs).to(self.device)
        entropies = torch.vstack(self.entropies).to(self.device)

        self.policy_optimzer.zero_grad()
        policy_loss = -(log_probs * gaes.detach() +
                        self.beta * entropies).mean()
        policy_loss.backward()
        self.policy_optimzer.step()

        self.value_optimzer.zero_grad()
        value_loss = gaes.pow(2).mean()
        value_loss.backward()
        self.value_optimzer.step()

    def learn(self, max_episodes, n_steps, average_len, target_reward, log=False):
        step = 0
        counter = 0
        rewards = []
        rewards_mean = []

        for episode in range(max_episodes):
            state, done = self.env.reset(), False
            reward_sum = 0
            self.reset()
            while not done:
                counter += 1
                step += 1
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.rewards.append(reward)
                state = next_state
                reward_sum += reward
                if counter % n_steps == 0 or done:
                    self.next_value(state, done)
                    self.optimize()
                    self.reset()

            rewards.append(reward_sum)
            if episode >= average_len:
                mean = np.mean(rewards[-average_len:])
                rewards_mean.append(mean)
                if log:
                    print('--------------------------------------------------------')
                    print(f'Episode: {episode}')
                    print(f'Step: {step}')
                    print(f'Train Mean: {mean}')
                    print('--------------------------------------------------------')
                    if mean >= target_reward:
                        print('GOAL ACHIEVED!')
                        self.pi.save()
                        return rewards_mean
        return rewards_mean
