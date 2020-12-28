import numpy as np


class ReplayBuffer():
    def __init__(self, state_dims, action_dims, max_memsize=10000, batch_size=64):
        self.batch_size = batch_size
        self.idx = 0
        self.max_memsize = max_memsize
        self.current_size = 0

        self.states = np.empty(shape=(max_memsize, state_dims))
        self.actions = np.empty(shape=(max_memsize, action_dims))
        self.rewards = np.empty(shape=(max_memsize, 1))
        self.next_states = np.empty(shape=(max_memsize, state_dims))
        self.terminals = np.empty(shape=(max_memsize, 1))

    def add_memory(self, state, action, reward, next_state, done):
        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_state
        self.terminals[self.idx] = done

        self.current_size = min(self.current_size+1, self.max_memsize)
        self.idx = (self.idx + 1) % self.max_memsize

    def sample(self):
        idxs = np.random.choice(len(self), self.batch_size, replace=False)

        states = self.states[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_states = self.next_states[idxs]
        terminals = self.terminals[idxs]

        return states, actions, rewards, next_states, terminals

    def __len__(self):
        return self.current_size
