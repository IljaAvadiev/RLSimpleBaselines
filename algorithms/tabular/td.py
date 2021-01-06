import numpy as np
from base import TabularAgent


###########################################################################################
# TD Tabular METHODS
###########################################################################################
class SarsaAgent(TabularAgent):
    '''
    On Policy TD Agent
    '''

    def __init__(self, nS, nA, alpha=0.01, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.01, epsilon_end=0.01):
        super(SarsaAgent, self).__init__(nS, nA, alpha, gamma,
                                         epsilon_start, epsilon_decay, epsilon_end)

    def learn(self, obs, action, next_obs, reward, done):
        next_action = self.act_eps_greedy(next_obs)
        target = reward + self.gamma * \
            self.Q[next_obs][next_action] * (not done)
        self.Q[obs][action] = self.Q[obs][action] + \
            self.alpha * (target - self.Q[obs][action])
        self.decrease_eps()


# Q-Learning
class QAgent(TabularAgent):
    '''
    Off Policy TD Agent
    '''

    def __init__(self, nS, nA, alpha=0.01, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.01, epsilon_end=0.01):
        super(QAgent, self).__init__(nS, nA, alpha, gamma,
                                     epsilon_start, epsilon_decay, epsilon_end)

        def learn(self, obs, action, next_obs, reward, done):
            next_action = self.act_greedy(next_obs)
            target = reward + self.gamma * \
                self.Q[next_obs][next_action] * (not done)
            self.Q[obs][action] = self.Q[obs][action] + \
                self.alpha * (target - self.Q[obs][action])
            self.decrease_eps()
