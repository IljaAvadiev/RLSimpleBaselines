import numpy as np


class EpisodicBuffer():
    def __init__(self, state_dims, action_dims, max_memory_size, batch_size, gamma, tau):
        self.batch_size = batch_size
        self.max_memory_size = max_memory_size
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.gamma = gamma
        self.tau = tau

    def reset(self):
        self.idx = 0
        self.start_idx = 0
        self.states = np.empty(
            shape=(self.max_memory_size, self.state_dims), dtype=np.float32)
        self.actions = np.empty(
            shape=(self.max_memory_size, self.action_dims), dtype=np.float32)
        self.rewards = np.empty(
            shape=(self.max_memory_size, 1), dtype=np.float32)
        self.logps = np.empty(
            shape=(self.max_memory_size, 1), dtype=np.float32)
        self.values = np.empty(
            shape=(self.max_memory_size, 1), dtype=np.float32)
        self.returns = np.empty(
            shape=(self.max_memory_size, 1), dtype=np.float32)
        self.gaes = np.empty(shape=(self.max_memory_size, 1), dtype=np.float32)

    def add_memory(self, state, action, reward, value, logp):
        assert self.idx < self.max_memory_size

        self.states[self.idx] = state
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.values[self.idx] = value
        self.logps[self.idx] = logp

        self.idx += 1

    def sample_batch(self):
        # the memory buffer has to be full
        assert self.idx == self.max_memory_size
        # the episode has to be finished
        assert self.idx == self.start_idx

        idxs = np.random.choice(self.max_memory_size,
                                self.batch_size, replace=False)

        return self.states[idxs], self.actions[idxs], self.returns[idxs], self.gaes[idxs], self.logps[idxs]

    def finish_episode(self, last_val=0):
        idxs = slice(self.start_idx, self.idx)
        rewards = np.append(self.rewards[idxs], last_val)
        values = np.append(self.values[idxs], last_val)

        # calculate returns
        returns = self.discount_cumsum(rewards, self.gamma)

        # calucalte gaes
        resisuals = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        gaes = self.discount_cumsum(resisuals, self.gamma * self.tau)

        self.returns[idxs] = returns[:-1]
        self.gaes[idxs] = gaes

        self.start_idx = self.idx

    def discount_cumsum(self, arr, rate):
        trajectory_len = len(arr)
        rates = np.array([rate**exp for exp in range(trajectory_len)])
        return np.array([np.sum(arr[i:] * rates[:trajectory_len-i])
                         for i in range(trajectory_len)]).reshape(-1, 1)
