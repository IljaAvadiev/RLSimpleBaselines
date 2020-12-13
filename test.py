import gym
from utils.helper import print_policy
from utils.helper import create_random_policy


env = gym.make('FrozenLake-v0')
nS = env.observation_space.n
nA = env.action_space.n
pi = create_random_policy(nS, nA)

print_policy(pi, nS, 4)
