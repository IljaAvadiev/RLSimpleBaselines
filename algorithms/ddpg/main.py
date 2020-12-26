import gym
from agent import Agent
import numpy as np

env = gym.make('LunarLanderContinuous-v2')
num_episodes = 10000
alpha = 0.0001
gamma = 0.99
tau = 0.995
state_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]
action_ranges = (env.action_space.low[0], env.action_space.high[0])
warmup = 10
mem_size = 10000
batch_size = 64
average_size = 100
win_reward = 200

ddpg_agent = Agent(state_dims, action_dims, action_ranges,
                   (400, 200), alpha, gamma, tau, mem_size, batch_size)
reward_tracking = []
reward_averaging = []
for episode in range(num_episodes):
    obs, done = env.reset(), False
    ddpg_agent.reset()
    while not done:
        if episode < warmup:
            action = env.action_space.sample()
        else:
            action = ddpg_agent.select_action(obs)
        new_obs, reward, done, _ = env.step(action)
        ddpg_agent.add_memory(obs, action, reward, new_obs, done)
        obs = new_obs

        if episode > warmup:
            ddpg_agent.optimize()

    if episode > warmup:
        reward_sum = 0
        obs, done = env.reset(), False
        while not done:
            action = ddpg_agent.online_policy.act(obs)
            new_obs, reward, done, _ = env.step(action)
            obs = new_obs
            reward_sum += reward
        reward_tracking.append(reward_sum)
        if len(reward_tracking) > average_size:
            average = np.mean(reward_tracking[-average_size:])
            print(f'Episode {episode - warmup} Reward {average}')
            if average >= win_reward:
                break
