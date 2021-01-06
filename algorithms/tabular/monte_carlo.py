import numpy as np

###########################################################################################
# MONTE CARLO METHODS
###########################################################################################


class MCAgent():
    '''
    First Visit Monte Carlo agent
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

    def act_greedy(self, state):
        action = np.argmax(self.Q[state])
        return action

    def act_eps_greedy(self, state):
        if np.random.rand() > self.epsilon:
            action = self.act_eps_greedy(state)
        else:
            action = np.random.choice(len(self.Q[state]))
        return action

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
