import random


def print_policy(pi,
                 nS,
                 n_cols,
                 name=None,
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
