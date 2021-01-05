import gym
import numpy as np
from agent import Agent

env = gym.make('LunarLanderContinuous-v2')


state_dims = env.observation_space.shape[0]
action_dims = env.action_space.shape[0]
max_action = env.action_space.high[0]
min_action = env.action_space.low[0]

n_games = 1500
hidden_dims = (400, 300)
lr_actor = 0.003
lr_critic = 0.003
alpha = 0.1
gamma = 0.99
tau = 0.0001
mem_size = 1000000
batch_size = 100
warm_up = 1000
win_condition = 200

sac_agent = Agent(state_dims, action_dims,
                  hidden_dims, lr_actor=lr_actor, lr_critic=lr_critic, alpha=alpha, gamma=gamma, tau=tau, mem_size=mem_size, batch_size=batch_size, log_sigma_min=0, log_sigma_max=3, warmup=warm_up)

track_reward = []
track_reward_mean = []
best_reward = -10000
for episode in range(n_games):
    obs, done = env.reset(), False

    # training
    while not done:
        action = sac_agent.choose_action(obs)
        new_obs, reward, done, _ = env.step(action)
        sac_agent.store_transition(obs, action, reward, new_obs, done)
        sac_agent.learn()
        obs = new_obs

    reward_sum = 0
    # evaluating
    obs, done = env.reset(), False
    while not done:
        action = sac_agent.greedy_action(obs)
        new_obs, reward, done, _ = env.step(action)
        obs = new_obs
        reward_sum += reward
    track_reward.append(reward_sum)
    print(f'Episode: {episode}, Reward: {reward_sum}')
    if (len(track_reward) > 100):
        mean = np.mean(track_reward[-100:])
        if mean > best_reward:
            best_reward == mean
        if mean > win_condition:
            break

        print(f'Episode: {episode}, Mean: {mean}')
