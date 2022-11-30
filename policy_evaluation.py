# here we implement policy iteration based on the simple gridworld run by explore.py
# using OpenAI gyms for this would pose some challenges at first, but the problem is simple enough to just be put together using numpy arrays
import os
import time
import numpy as np

os.system('cls' if os.name == 'nt' else 'clear')
test_grid_size = int(input('Enter grid size to use: '))

# even though not being used at the moment, for generality we're defining the policy as being a function of an action and a current state
def test_policy(action, state):
    return 0.25




# this class codifies all the dynamics of the problem: a simple gridworld, with the target in the lower-right corner.
# as a starter in implementing GridWorld stochastics, I will try to program simple stochastics into this gridworld's dynamics
# E.G., given a certain action in a rectangular direction, we can assign the successor state some stochastics like
# 85% probability of ending up where you expected, and 5% in each of the 3 other directions.
class MarkovGridWorld():
    def __init__(self, grid_size=3, discount_factor=1):
        self.grid_size = grid_size # keep this unchanged, things are mostly hardcoded at the moment
        self.action_space = (0,1,2,3)
        self.discount_factor = discount_factor
        self.rewards = -1 * np.ones([grid_size, grid_size])
        self.terminal_state = np.array([grid_size-1, grid_size-1]) # terminal state in the bottom right corner
        self.rewards[tuple(self.terminal_state)] = 0
        self.action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    # this is where the actual DYNAMICS live
    def state_transition(self, state, action):
        direction = self.action_to_direction[action]
        new_state = np.clip(
            state + direction, 0, self.grid_size - 1
        )
        # override the above if we were actually in the terminal state!
        if np.array_equal(state, self.terminal_state):
            new_state = self.terminal_state
        return new_state, self.rewards[tuple(new_state)]


# epsilon = the threshold delta must go below in order for us to stop
def policy_evaluation(policy, MDP, epsilon=0.2, max_iterations=20):
    current_value = np.zeros([MDP.grid_size, MDP.grid_size])
    change = np.zeros([MDP.grid_size, MDP.grid_size]) # this will store the change in the value for each state, in the latest iteration
    delta = 0 # initialising the variable that will store the max change in the value_function across all states
    iteration_no = 1
    while (delta == 0 or delta > epsilon) and iteration_no <= max_iterations:
        print(f'Iteration number: {iteration_no}')
        print(f'Max iterations: {max_iterations}')
        print(f'Epsilon: {epsilon}')
        print()
        print('Current value function estimate:')
        print(current_value)
        print()

        for row in range(MDP.grid_size):
            for col in range(MDP.grid_size):
                state = np.array([row,col])
                old_state_value = current_value[tuple(state)]
                
                current_value_update = 0
                # using deterministic MDP where an action from a state fully determines the successor state here!
                for action in MDP.action_space:
                    successor_state, reward = MDP.state_transition(state, action)
                    current_value_update += policy(action, state) * (reward + MDP.discount_factor * current_value[tuple(successor_state)])
                current_value[tuple(state)] = current_value_update
                
                change[tuple(state)] = current_value[tuple(state)] - old_state_value
        delta = change.max()
        print('Changes to value function estimate:')
        print(change)
        print()
        print()
        print()
        print()
        time.sleep(0.3)
        iteration_no += 1
    return current_value





value = policy_evaluation(policy = test_policy, MDP = MarkovGridWorld(grid_size = test_grid_size))
print('Final value function estimate:')
print(value)
print()