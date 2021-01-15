from algorithms.deeprl.ppo.model import Pi, V, DEVICE
from algorithms.deeprl.ppo.memory import EpisodicBuffer
import torch
import numpy as np


class PPO():
    def __init__(self, env, state_dims, action_dims, hidden_dims, activation, optimizer,
                 pi_alpha, v_alpha, beta, gamma, tau, clip_ratio, horizon, epochs, batch_size, dir, name):
        self.env = env
        self.beta = beta
        self.memory = EpisodicBuffer(
            state_dims, 1, horizon, batch_size, gamma, tau)
        self.clip_ratio = clip_ratio
        self.horizon = horizon
        self.epochs = epochs
        self.pi = Pi(state_dims, action_dims,
                     hidden_dims, activation, dir, name)
        self.v = V(state_dims, hidden_dims, activation, dir, name)
        self.device = DEVICE
        self.policy_optimzer = optimizer(self.pi.parameters(), pi_alpha)
        self.value_optimzer = optimizer(self.v.parameters(), v_alpha)

    def reset(self):
        self.memory.reset()

    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).to(self.device)
            action, log_prob, _ = self.pi(state)
            value = self.v(state)
            action = action.cpu().detach().item()
            log_prob = log_prob.cpu().detach().item()
            value = value.cpu().detach().item()
            return action, log_prob, value

    def get_batch(self):
        states, actions, returns, gaes, logps = self.memory.sample_batch()
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        returns = torch.tensor(returns, dtype=torch.float32).to(self.device)
        gaes = torch.tensor(gaes, dtype=torch.float32).to(self.device)
        logps = torch.tensor(logps, dtype=torch.float32).to(self.device)

        return states, actions, returns, gaes, logps

    def optimize(self):
        for _ in range(self.epochs):
            states, actions, returns, gaes, old_logps = self.get_batch()

            # policy loss
            _, log_probs, entropies = self.pi(states, actions)
            log_probs = log_probs.reshape(-1, 1)
            ratios = (log_probs - old_logps).exp()

            clip_adv = torch.clamp(
                ratios, 1-self.clip_ratio, 1+self.clip_ratio) * gaes

            self.policy_optimzer.zero_grad()
            policy_loss = -(torch.min(ratios * gaes, clip_adv) +
                            self.beta * entropies).mean()
            policy_loss.backward()
            self.policy_optimzer.step()

            # value loss
            self.value_optimzer.zero_grad()
            value_loss = ((self.v(states) - returns) ** 2).mean()
            value_loss.backward()
            self.value_optimzer.step()

    def learn(self, max_episodes, average_len, target_reward, log=False):
        step = 0
        counter = 0
        rewards = []
        rewards_mean = []

        self.reset()
        for _ in range(max_episodes):
            state, done = self.env.reset(), False
            while not done:
                counter += 1
                step += 1
                action, log_prob, value = self.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.add_memory(state, action, reward, value, log_prob)
                state = next_state

                if done:
                    self.memory.finish_episode()
                if counter % self.horizon == 0:
                    counter = 0
                    _, _, last_value = self.act(state)
                    self.memory.finish_episode(last_value)
                    self.optimize()
                    self.reset()
                    reward_sum = self.evaluate()
                    rewards.append(reward_sum)

            if len(rewards) >= average_len:
                mean = np.mean(rewards[-average_len:])
                rewards_mean.append(mean)
                if log:
                    print('--------------------------------------------------------')
                    print(f'Episode: {len(rewards_mean)}')
                    print(f'Step: {step}')
                    print(f'Evaluate Mean: {mean}')
                    print('--------------------------------------------------------')
                    if mean >= target_reward:
                        print('GOAL ACHIEVED!')
                        self.pi.save()
                        return rewards_mean
        return rewards_mean

    def evaluate(self):
        state, done = self.env.reset(), False
        reward_sum = 0

        while not done:
            action, _, _ = self.act(state)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            reward_sum += reward

        return reward_sum
