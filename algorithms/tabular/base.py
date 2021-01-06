###########################################################################################
# Base Class for tabular RL agents
###########################################################################################
import numpy as np


class TabularAgent():
    '''
    Base Class for all tabular reinforcement learning agents
    '''

    def __init__(self, nS, nA, alpha=0.01, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.01, epsilon_end=0.01):
        self.Q = np.zeros((nS, nA))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.nS = nS
        self.nA = nA

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
