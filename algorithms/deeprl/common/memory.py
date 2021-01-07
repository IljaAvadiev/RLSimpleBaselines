import numpy as np


class ReplayBuffer():
    '''
    A class for storing and retrieving experiences
    '''

    def __init__(self, state_dims, action_dims, max_memory_size, batch_size):
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.idx = 0
        self.current_memory_size = 0

        self.states = np.empty(
            shape=(max_memory_size, state_dims), dtype=np.float32)
        self.actions = np.empty(
            shape=(max_memory_size, action_dims), dtype=np.float32)
        self.next_states = np.empty(
            shape=(max_memory_size, state_dims), dtype=np.float32)
        self.rewards = np.empty(shape=(max_memory_size, 1), dtype=np.float32)
        self.terminals = np.empty(shape(max_memory_size, 1), dtype=np.bool)

    def add_memory(self, state, action, next_state, reward, done):
        self.states[idx] = state
        self.actions[idx] = action
        self.next_states[idx] = next_state
        self.rewards[idx] = reward
        self.terminals[idx] = done

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
