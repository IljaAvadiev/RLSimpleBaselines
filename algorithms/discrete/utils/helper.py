import random
import numpy as np


def print_policy(pi,
                 nS,
                 n_cols,
                 name='FrozenLake',
                 terminal_states=[5, 7, 11, 12, 15],
                 actions_mapping={0: '\u2190', 1: '\u2193', 2: '\u2192', 3: '\u2191'}):
    '''
    Prints a policy for a gridworld

    Args:
        pi:               Policy
        nS:               Number of states in the gridworld
        n_cols:           Number of columns in the gridworld
        name:             Name of the policy
        terminal_states:  List of all states that lead to itself with 100%
        actions_mapping:  List of symbols to print instead of integers

    Returns: 
        None
    '''
    if name is not None:
        print('\n')
        print('\033[1m' + name + '\033[0m')
        print('\n')

    for state in range(nS):
        end = '\n' if (state + 1) % n_cols == 0 else ' '

        if state in terminal_states:
            print('\u25A0'.rjust(10), end=end)
            continue

        action = pi(state)
        print(actions_mapping[action].rjust(10), end=end)


def create_random_policy(nS, nA, seed=42):
    '''
    Generates a random policy for a gridworld

    Args:
        nS:     Number of states in a gridworld
        nA:     Number of actions in a state
        seed:   Seed for the random module

    Returns:
        random_policy: A function generating an action given a state
    '''

    random.seed(seed)
    policy = {}
    for i in range(nS):
        policy[i] = random.randint(0, nA-1)

    def random_policy(s):
        return policy[s]

    return random_policy


def print_state_value_func(V, n_cols, name=None):
    '''
    Prints a state value function of a grid world

    Args:
        V:       state value function of a grid world
        n_cols:  number of columns in the grid world
    Returns: 
        None
    '''

    if name is not None:
        print('\n')
        print('\033[1m' + name + '\033[0m')
        print('\n')

    for state in range(len(V)):
        end = '\n' if (state + 1) % n_cols == 0 else ' '
        value = V[state]
        print(f'{value:.5f}', end=end)


def generate_episode(env, pi):
    '''
    Generates an episode following a policy pi.
    An episode is a succession of experiences until the terminal state. 
    A tuple of state, action, reward, next_state and done constitute an exprience.

    Args: 
        env: OpenAI gym environment to interract with
        pi:  Policy that is used to create an episode

    Returns:
        episode
    '''
    episode = []
    obs, done = env.reset(), False
    while not done:
        action = pi(obs)
        new_obs, reward, done, _ = env.step(action)
        experience = (obs, action, reward, new_obs, done)
        episode.append(experience)
        obs = new_obs

    return episode


def linear_decay(start_value, end_value, decay_duration, full_duration):
    '''
    Generates a list of float values that decrease from a start_value to the end_value in a linear fashion

    Args:
        start_value:      The starting value for a float value
        end_value:        The lowest value that is allowed
        decay_duration:   Number of episodes the value is decreasing
        full_duration:    Number of episodes the value is needed

    Returns:
        decay_values:     Numpy array of decaying values
    '''

    decay_values = np.arange(decay_duration)[::-1]
    decay_values = decay_values / np.max(decay_values)

    values_range = start_value - end_value
    decay_values = decay_values * values_range + end_value

    decay_values = np.pad(
        decay_values, (0, full_duration - decay_duration), mode='edge')

    return decay_values


def epsilon_greedy(Q, state, epsilon):
    '''
    Selects an action using epsillon-greedy policy

    Args: 
        Q:       Action Value function of a policy pi
        state:   State in the MDP context
        epsilon: Probability with which a random action is selected

    Returns:
        Action
    '''

    if np.random.rand() > epsilon:
        action = np.argmax(Q[state])
    else:
        action = np.random.randint(len(Q[state]))

    return action
