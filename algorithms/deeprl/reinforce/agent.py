from algorithms.deeprl.reinforce.model import Pi, DEVICE
import torch
import numpy as np


class Reinforce():
    def __init__(self, env, state_dims, action_dims, hidden_dims, activation, optimizer,
                 alpha, gamma, dir, name):
        self.env = env
        self.gamma = gamma
        self.pi = Pi(state_dims, action_dims,
                     hidden_dims, activation, dir, name)
        self.device = DEVICE
        self.optimzer = optimizer(self.pi.parameters(), alpha)

    def reset(self):
        self.log_probs = []
        self.rewards = []

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action, log_prob = self.pi(state)
        self.log_probs.append(log_prob)
        return action.cpu().detach().item()

    def optimize(self):
        trajectory_len = len(self.rewards)
        gammas = [self.gamma**exp for exp in range(trajectory_len)]
        returns = np.array([np.sum(np.array(self.rewards[i:]) * np.array(gammas[:trajectory_len-i]))
                            for i in range(trajectory_len)])

        log_probs = torch.vstack(self.log_probs).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(
            self.device).view(-1, 1)

        self.optimzer.zero_grad()
        loss = -(log_probs * returns).mean()
        loss.backward()
        self.optimzer.step()

    def learn(self, max_episodes, average_len, target_reward, log=False):
        step = 0
        rewards = []
        rewards_mean = []

        for episode in range(max_episodes):
            state, done = self.env.reset(), False
            reward_sum = 0
            self.reset()
            while not done:
                step += 1
                action = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.rewards.append(reward)
                state = next_state
                reward_sum += reward
            self.optimize()

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
                        break
