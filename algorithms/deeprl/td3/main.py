import gym
import numpy as np
from agent import Agent

env = gym.make('BipedalWalker-v3')


state_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]

n_games = 1500
hidden_dims = (400, 300)
lr_actor = 0.001
lr_critic = 0.001
gamma = 0.99
tau = 0.005
mem_size = 1000000
batch_size = 64
warm_up = 1000
win_condition = 275

td3_agent = Agent(state_dims, action_dims,
                  hidden_dim1=hidden_dims[0], hidden_dim2=hidden_dims[1], lr_actor=lr_actor, lr_critic=lr_critic, gamma=gamma, tau=tau, mem_size=mem_size, batch_size=batch_size, max_action=max_action, min_action=min_action, warmup=warm_up)

track_reward = []
track_reward_mean = []
best_reward = -10000
for episode in range(n_games):
    obs, done = env.reset(), False

    # training
    while not done:
        action = td3_agent.choose_action(obs)
        new_obs, reward, done, _ = env.step(action)
        td3_agent.store_transition(obs, action, reward, new_obs, done)
        td3_agent.learn()
        obs = new_obs

    reward_sum = 0
    # evaluating
    obs, done = env.reset(), False
    while not done:
        action = td3_agent.greedy_action(obs)
        new_obs, reward, done, _ = env.step(action)
        obs = new_obs
        reward_sum += reward
    track_reward.append(reward_sum)
    print(f'Episode: {episode}, Reward: {reward_sum}')
    if (len(track_reward) > 100):
        mean = np.mean(track_reward[-100:])
        if mean > best_reward:
            best_reward == mean
            td3_agent.save()
        if mean > win_condition:
            td3_agent.save()
            break

        print(f'Episode: {episode}, Mean: {mean}')
