import numpy as np
from numpy.core.fromnumeric import shape


class ReplayBuffer():
    '''
    A class for storing and retrieving experiences
    '''

    def __init__(self, state_dims, action_dims, max_memory_size, batch_size):
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size
        self.idx = 0
        self.current_memory_size = 0

        self.states = np.empty(
            shape=(max_memory_size, *state_dims), dtype=np.float32)
        self.actions = np.empty(
            shape=(max_memory_size, action_dims), dtype=np.float32)
        self.next_states = np.empty(
            shape=(max_memory_size, *state_dims), dtype=np.float32)
        self.rewards = np.empty(shape=(max_memory_size, 1), dtype=np.float32)
        self.terminals = np.empty(shape=(max_memory_size, 1), dtype=np.bool)

    def add_memory(self, state, action, next_state, reward, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.next_states[self.idx] = next_state
        self.rewards[self.idx] = reward
        self.terminals[self.idx] = done

        self.idx = (self.idx + 1) % self.max_memory_size
        self.current_memory_size = min(
            self.current_memory_size + 1, self.max_memory_size)

    def sample_batch(self):
        idxs = np.random.choice(len(self), self.batch_size, replace=False)

        states = self.states[idxs]
        actions = self.actions[idxs]
        next_states = self.next_states[idxs]
        rewards = self.rewards[idxs]
        terminals = self.terminals[idxs]

        return states, actions, next_states, rewards, terminals

    def __len__(self):
        return self.current_memory_size


class PER():
    '''
    A class for storing and retrieving experiences
    '''

    def __init__(self, state_dims, action_dims, max_memory_size, batch_size, alpha, beta, beta_increment, epsilon):
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size
        self.idx = 0
        self.current_memory_size = 0

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.states = np.zeros(
            shape=(max_memory_size, *state_dims), dtype=np.float32)
        self.actions = np.zeros(
            shape=(max_memory_size, action_dims), dtype=np.float32)
        self.next_states = np.zeros(
            shape=(max_memory_size, *state_dims), dtype=np.float32)
        self.rewards = np.zeros(shape=(max_memory_size, 1), dtype=np.float32)
        self.terminals = np.zeros(shape=(max_memory_size, 1), dtype=np.bool)
        self.priorities = np.zeros(
            shape=(max_memory_size, 1), dtype=np.float64)

    def add_memory(self, state, action, next_state, reward, done):
        if self.current_memory_size == 0:
            priority = 1
        else:
            priority = self.priorities.max()

        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.next_states[self.idx] = next_state
        self.rewards[self.idx] = reward
        self.terminals[self.idx] = done
        self.priorities[self.idx] = priority

        self.idx = (self.idx + 1) % self.max_memory_size
        self.current_memory_size = min(
            self.current_memory_size + 1, self.max_memory_size)

    def anneal_beta(self):
        self.beta = min(1, self.beta + self.beta_increment)

    def update_priority(self, idxs, td_errors):
        self.priorities[idxs] = np.abs(td_errors) + self.epsilon

    def sample_batch(self):
        priorities = self.priorities[:self.current_memory_size]
        probs = priorities**self.alpha / \
            np.sum(priorities**self.alpha)

        weights = (len(self) * probs)**-self.beta
        idxs = np.random.choice(
            len(self), self.batch_size, replace=False, p=np.squeeze(probs, axis=1))

        states = self.states[idxs]
        actions = self.actions[idxs]
        next_states = self.next_states[idxs]
        rewards = self.rewards[idxs]
        terminals = self.terminals[idxs]
        weights = weights[idxs]

        self.anneal_beta()
        return states, actions, next_states, rewards, terminals, idxs, weights

    def __len__(self):
        return self.current_memory_size
