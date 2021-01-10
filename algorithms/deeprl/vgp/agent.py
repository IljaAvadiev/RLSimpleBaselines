from algorithms.deeprl.vgp.model import Pi, V, DEVICE
import torch
import numpy as np


class VPG():
    def __init__(self, env, state_dims, action_dims, hidden_dims, activation, optimizer,
                 pi_alpha, v_alpha, beta, gamma, dir, name):
        self.env = env
        self.beta = beta
        self.gamma = gamma
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

    def optimize(self):
        trajectory_len = len(self.rewards)
        gammas = [self.gamma**exp for exp in range(trajectory_len)]
        returns = np.array([np.sum(np.array(self.rewards[i:]) * np.array(gammas[:trajectory_len-i]))
                            for i in range(trajectory_len)])

        log_probs = torch.vstack(self.log_probs).to(self.device)
        values = torch.vstack(self.values).to(self.device)
        entropies = torch.vstack(self.entropies).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(
            self.device).view(-1, 1)

        advantages = returns - values

        self.policy_optimzer.zero_grad()
        policy_loss = -(log_probs * advantages.detach() +
                        self.beta * entropies).mean()
        policy_loss.backward()
        self.policy_optimzer.step()

        self.value_optimzer.zero_grad()
        value_loss = advantages.pow(2).mean()
        value_loss.backward()
        self.value_optimzer.step()

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
                        return rewards_mean
        return rewards_mean
