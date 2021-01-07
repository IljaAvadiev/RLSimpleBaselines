import numpy as np
from algorithms.tabular.base import TabularAgent
###########################################################################################
# MONTE CARLO Tabular METHODS
###########################################################################################


class MCAgent(TabularAgent):
    '''
    First Visit Monte Carlo agent
    '''

    def __init__(self, nS, nA, alpha=0.01, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.01, epsilon_end=0.01):
        super(MCAgent, self).__init__(nS, nA, alpha, gamma,
                                      epsilon_start, epsilon_decay, epsilon_end)

    def learn(self, episode):
        visited = np.zeros((self.nS, self.nA), dtype=np.bool)
        for step, (obs, action, _, _, _) in enumerate(episode):
            # first visit
            if visited[obs][action] == True:
                continue
            visited[obs][action] = True

            remaining_rewards = np.array(episode)[step:, 3]
            remaining_steps = len(remaining_rewards)

            gammas = np.array(
                [self.gamma**exp for exp in range(remaining_steps)])
            target = np.sum(remaining_rewards * gammas)
            self.Q[obs][action] = self.Q[obs][action] + \
                self.alpha * (target - self.Q[obs][action])

            self.decrease_eps()
