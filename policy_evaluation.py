# here we implement policy iteration based on the simple gridworld run by explore.py
# using OpenAI gyms for this would pose some challenges at first, but the problem is simple enough to just be put together using numpy arrays
import os
import time
import numpy as np

os.system('cls' if os.name == 'nt' else 'clear')
#test_grid_size = int(input('Enter grid size to use: '))

# even though not being used at the moment, for generality we're defining the policy as being a function of an action and a current state
def test_policy(action, state):
    return 0.25




# this class codifies all the dynamics of the problem: a simple gridworld, with the target in the lower-right corner.
# as a starter in implementing GridWorld stochastics, I will try to program simple stochastics into this gridworld's dynamics
# E.G., given a certain action in a rectangular direction, we can assign the successor state some stochastics like
# 85% probability of ending up where you expected, and 5% in each of the 3 other directions.
# the argument direction_probability controls the probability with which we can expect the agent's selected action to result in the 'expected' successor state.
class MarkovGridWorld():
    def __init__(self, grid_size=3, discount_factor=1, direction_probability = 1):
        self.grid_size = grid_size # keep this unchanged, things are mostly hardcoded at the moment
        self.action_space = (0,1,2,3)
        self.discount_factor = discount_factor
        #self.rewards = -1 * np.ones([grid_size, grid_size])
        self.terminal_state = np.array([grid_size-1, grid_size-1]) # terminal state in the bottom right corner
        #self.rewards[tuple(self.terminal_state)] = 0
        self.action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        self.rng = np.random.default_rng() # construct a default numpy random number Generator class instance, to use in stochastics
        self.direction_probability = direction_probability
        # for ease of iterating over all states, define a 2 x (grid_size**2) matrix below
        self.state_space = np.zeros(shape=(self.grid_size**2,2))
        state_counter = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                self.state_space[state_counter, :] = [row, col]
                state_counter += 1


    def reward(self, state):
        if np.array_equal(state, self.terminal_state):
            return 0
        else:
            return -1


    # environment dynamics give the probability of a successor state, given a current state and an action
    # crucial to define these in this generalised form, in order to implement general policy evaluation algorithm
    def environment_dynamics(self, successor_state, current_state, action):
        # first, consider the case where the current state is the terminal state
        if np.array_equal(current_state, self.terminal_state):
            if np.array_equal(successor_state, self.terminal_state):
                return 1
            else:
                return 0
        
        prob_other_directions = (1 - self.direction_probability) / 3
        # the stochastics array describes the probability, given an action from (0,1,2,3), of the result corresponding to what we'd expect from each of those actions
        # if action == 1, for example, if stochastics == array[0.1,0.7,0.1,0.1], then the resulting successor state will be what we would expect of action == 1 with 70% probability,
        # and 10% probability for each of the other directions 
        stochastics = np.ones(4) * prob_other_directions
        stochastics[action] = self.direction_probability
        state_difference = successor_state - current_state # sort of a vector from current state to successor_state
        # if the successor_state is reachable from current_state, we return the probabilities of getting there, given our input action
        # these probabilities have been defined by the stochastics vector above
        for action_int in range(4):
            if np.array_equal(state_difference, self.action_to_direction[action_int]):
                return stochastics[action_int]
        else:
            return 0




    # this is where the actual DYNAMICS live
    def state_transition(self, state, action):
        # the stochastics array describes the probability, given an action from (0,1,2,3), of the result corresponding to what we'd expect from each of those actions
        # if action == 1, for example, if stochastics == array[0.1,0.7,0.1,0.1], then the resulting successor state will be what we would expect of action == 1 with 70% probability,
        # and 10% probability for each of the other directions 
        prob_other_directions = (1 - self.direction_probability) / 3
        stochastics = np.ones(4) * prob_other_directions
        stochastics[action] = self.direction_probability
        effective_action = self.rng.choice(4, p=stochastics) # this is the effective action, after sampling from the given-action-biased distribution. Most times this will be equal to action
        effective_direction = self.action_to_direction[effective_action]
        new_state = np.clip(
            state + effective_direction, 0, self.grid_size - 1
        )
        # override the above if we were actually in the terminal state!
        if np.array_equal(state, self.terminal_state):
            new_state = self.terminal_state
        return new_state, self.reward(new_state)


# epsilon = the threshold delta must go below in order for us to stop
def policy_evaluation(policy, MDP, epsilon=0, max_iterations=20):
    print(MDP.state_space)

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

        # below 3 lines effectively equivalent to 'for state in state space of MDP'
        for state_number in range(MDP.grid_size ** 2):
            state = MDP.state_space[state_number,:]
            
            old_state_value = current_value[state.astype(int)]
            
            current_value_update = 0
            for action in MDP.action_space:
                sub_sum = 0
                for successor_state_number in range(MDP.grid_size ** 2):
                    successor_state = MDP.state_space[successor_state_number]
                    sub_sum += MDP.environment_dynamics(successor_state, state, action) * (MDP.reward(successor_state) + MDP.discount_factor * current_value[successor_state.astype(int)])
                    if np.array_equal(state, MDP.terminal_state):
                        print(MDP.environment_dynamics(successor_state, state, action))
                current_value_update += policy(action,state) * sub_sum

            current_value[state.astype(int)] = current_value_update
            
            change[state.astype(int)] = abs(current_value[state.astype(int)] - old_state_value)
        delta = change.max()
        print('Absolute changes to value function estimate:')
        print(change)
        print()
        print()
        print()
        print()
        time.sleep(0.3)
        iteration_no += 1
    return current_value



MDP = MarkovGridWorld(grid_size = 3)
print(MDP.environment_dynamics([2,1], [2,2], 3))
print(MDP.reward([2,1]))

#value = policy_evaluation(policy = test_policy, MDP = MarkovGridWorld(grid_size = test_grid_size))
#print('Final value function estimate:')
#print(value)
#print()