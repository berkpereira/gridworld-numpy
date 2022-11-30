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

    def reward(self, state):
        if state == self.terminal_state:
            return 0
        else:
            return -1


    # environment dynamics give the probability of a successor state, given a current state and an action
    # crucial to define these in this generalised form, in order to implement general policy evaluation algorithm
    def environment_dynamics(self, successor_state, current_state, action):
        prob_other_directions = (1 - self.direction_probability) / 3
        # the stochastics array describes the probability, given an action from (0,1,2,3), of the result corresponding to what we'd expect from each of those actions
        # if action == 1, for example, if stochastics == array[0.1,0.7,0.1,0.1], then the resulting successor state will be what we would expect of action == 1 with 70% probability,
        # and 10% probability for each of the other directions 
        stochastics = np.ones(4) * prob_other_directions
        stochastics[action] = self.direction_probability
        state_difference = successor_state - current_state # sort of a vector from current state to successor_state
        # if the successor_state is reachable from current_state, we return the probabilities of getting there, given our input action
        # these probabilities have been defined by the stochastics vector above
        match state_difference:
            case self.action_to_direction[0]:
                return stochastics[0]
            case self.action_to_direction[1]:
                return stochastics[1]
            case self.action_to_direction[2]:
                return stochastics[2]
            case self.action_to_direction[3]:
                return stochastics[3]
            case _: # successor_state is outside of the domain reachable from the current_state (might be successor_state == current_state !)
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