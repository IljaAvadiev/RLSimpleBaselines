import numpy as np


# SARSA
class SarsaAgent():
    '''
    On Policy TD Agent
    '''

    def __init__(self, nS, nA, alpha=0.01, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.01, epsilon_end=0.01):
        self.Q = np.zeros((nS, nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end

    def decrease_eps(self):
        self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end

    def act_greedy(self, obs):
        action = np.argmax(self.Q[obs])
        return action

    def act_eps_greedy(self, obs):
        if np.random.rand() > self.epsilon:
            action = self.act_greedy(obs)
        else:
            action = np.random.choice(len(self.Q[obs]))
        return action

    def learn(self, obs, action, next_obs, reward, done):
        if reward > 0.0:
            print(reward)
        next_action = self.act_eps_greedy(next_obs)
        target = reward + self.gamma * \
            self.Q[next_obs][next_action] * (not done)
        self.Q[obs][action] = self.Q[obs][action] + \
            self.alpha * (target - self.Q[obs][action])
        self.decrease_eps()


# Q-Learning
class QAgent():
    '''
    Off Policy TD Agent
    '''

    def __init__(self, nS, nA, alpha=0.01, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.01, epsilon_end=0.01):
        self.Q = np.zeros((nS, nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end

    def decrease_eps(self):
        self.epsilon -= self.epsilon_decay
        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end

    def act_greedy(self, obs):
        action = np.argmax(self.Q[obs])
        return action

    def act_eps_greedy(self, obs):
        if np.random.rand() > self.epsilon:
            action = self.act_greedy(obs)
        else:
            action = np.random.choice(len(self.Q[obs]))
        return action

    def learn(self, obs, action, next_obs, reward, done):
        next_action = self.act_greedy(next_obs)
        target = reward + self.gamma * \
            self.Q[next_obs][next_action] * (not done)
        self.Q[obs][action] = self.Q[obs][action] + \
            self.alpha * (target - self.Q[obs][action])
        self.decrease_eps()
