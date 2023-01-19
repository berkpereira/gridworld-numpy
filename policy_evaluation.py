# here we implement policy iteration based on the simple gridworld run by explore.py
# using OpenAI gyms for this would pose some challenges at first, but the problem is simple enough to just be put together using numpy arrays
import os
import numpy as np

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
        self.terminal_state = np.array([grid_size-1, grid_size-1]) # terminal state in the bottom right corner
        self.action_to_direction = {
            0: np.array([1, 0]), # down
            1: np.array([0, 1]), # right
            2: np.array([-1, 0]), # up
            3: np.array([0, -1]), # left
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

    def direction_to_action(self, direction):
        for action in range(4):
            if np.array_equal(direction, self.action_to_direction[action]):
                return action

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
        # if the successor_state is reachable from current_state, we return the probabilities of getting there, given our input action
        # these probabilities have been defined by the stochastics vector above
        
        
        # HERE LIES AN ISSUE. NOT CONSIDERING CLIPPING THE ARRAY AT THE BOUNDARIES OF THE DOMAIN!!!
        successor_probability = 0 # initialise probability of successor, might in the end be sum of various components of stochastics vector
        for direction_number in range(4):
            direction = self.action_to_direction[direction_number]
            if np.array_equal(np.clip(current_state + direction, 0, self.grid_size - 1), successor_state):
                successor_probability += stochastics[direction_number]
        return successor_probability




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
    current_value = np.zeros([MDP.grid_size, MDP.grid_size])
    change = np.zeros([MDP.grid_size, MDP.grid_size]) # this will store the change in the value for each state, in the latest iteration
    delta = 0 # initialising the variable that will store the max change in the value_function across all states
    iteration_no = 1
    while (delta == 0 or delta > epsilon) and iteration_no <= max_iterations:
        print(f'Iteration number: {iteration_no}')
        print()
        print('Current value function estimate:')
        print(current_value)
        print()
        # below 3 lines effectively equivalent to 'for state in state space of MDP'
        for state in MDP.state_space:
            old_state_value = current_value[tuple(state.astype(int))]
            current_value_update = 0
            for action in MDP.action_space:
                sub_sum = 0
                for successor_state in MDP.state_space:
                    sub_sum += MDP.environment_dynamics(successor_state, state, action) * (MDP.reward(successor_state) + MDP.discount_factor * current_value[tuple(successor_state.astype(int))])
                current_value_update += policy(action,state) * sub_sum
            current_value[tuple(state.astype(int))] = current_value_update
            change[tuple(state.astype(int))] = abs(current_value[tuple(state.astype(int))] - old_state_value)
        delta = change.max()
        print('Absolute changes to value function estimate:')
        print(change)
        print()
        print()
        print()
        print()
        iteration_no += 1
    return current_value

# the below returns whether successor_state is in principle reachable from current_state, given the gridworld assumption
# of a single rectangular move per time step in the grid environment domain
def is_accessible(current_state, successor_state):
    state_difference = successor_state - current_state
    if np.array_equal(state_difference, [1,0]) or np.array_equal(state_difference, [-1,0]) or np.array_equal(state_difference, [0,1]) or np.array_equal(state_difference, [0,-1]):
        return True
    else:
        return False

# this function returns all states that are accessible from the current_state of the agent
# since this is intended for generating greedy policies and other useful stuff, we'll rule out the agent's own state, even when that is accessible (e.g., by trying to move outside of the domain boundaries)
def accessible_states(current_state, MDP):
    output = np.array([])
    action_space_size = len(MDP.action_space)
    for action in range(action_space_size):
        direction = MDP.action_to_direction[action]
        potential_accessible = np.clip(current_state + direction, 0, MDP.grid_size - 1) 
        if not np.array_equal(potential_accessible, current_state):
            if output.size == 0:
                output = np.array([potential_accessible])
            else:
                output = np.row_stack((output, potential_accessible))
    return output

# at the moment, this actually returns a 2D array with integers codifying greedy actions in it, with respect to an input value function
# from this, still need to construct a policy as a function policy(action, state), which returns a probability distribution over actions, given some current state
# keep in mind that such a greedy policy will always be deterministic, so the probability distribution will be very boring, with 1 assigned to the greedy action and 0 elsewhere
# however, this general structure is useful as it can be used directly in the generalised policy evaluation algorithm we've implemented, which assumes that form of a policy(action, satte)
def greedy_policy(value_function, MDP):
    policy_array = np.empty(shape=(MDP.grid_size, MDP.grid_size)) # this array stores actions (0,1,2,3) which codify the greedy policy
    for state in MDP.state_space:
        potential_next_states = accessible_states(state, MDP)
        max_next_value = np.NINF # initialise max value attainable as minus infinity
        for successor_state in potential_next_states:
            potential_value = value_function[tuple(successor_state.astype(int))]
            if potential_value > max_next_value:
                greedy_direction = successor_state - state
                max_next_value = potential_value
        policy_array[tuple(state.astype(int))] = MDP.direction_to_action(greedy_direction)
    return policy_array

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    default = input('Run policy evaluation with default parameters? (y/n) ')
    if default.split()[0][0].upper() == 'Y':
        grid_size = 3
        direction_probability = 1
        max_iterations = 20
        epsilon = 0
    else:
        grid_size = int(input('Input grid size: '))
        direction_probability = float(input('Input probability of action success: '))
        max_iterations = int(input('Input max number of iterations: '))
        epsilon = float(input('Input epsilon for convergence: '))

    GridWorld = MarkovGridWorld(grid_size=grid_size, direction_probability=direction_probability)
    print('-----------------------------------------------------------------------------')
    print('Running policy evaluation.')
    print(f'Grid size: {GridWorld.grid_size}')
    print(f'Terminal state: {GridWorld.terminal_state}')
    print(f'Discount factor: {GridWorld.discount_factor}')
    print(f'Probability of action resulting in intended direction of motion: {GridWorld.direction_probability}')
    print(f'Epsilon: {epsilon}')
    print(f'Max iterations: {max_iterations}')
    print('-----------------------------------------------------------------------------')
    input('Press Enter to continue...')
    print()
    value = policy_evaluation(policy = test_policy, MDP = GridWorld, epsilon = epsilon, max_iterations=max_iterations)
    greedy_policy_array = greedy_policy(value, GridWorld)
    print('-----------------------------------------------------------------------------')
    print()
    print()
    print()
    print()
    print('Final value estimation:')
    print(value)
    print()
    print('Greedy policy ARRAY (NOT ACTUAL policy(action,state) JUST YET, NEED TO ADAPT THAT) with respect to final value function estimate:')
    print(greedy_policy_array)


if __name__ == "__main__":
    import cProfile
    cProfile.run('main()', 'output.dat')

    import pstats
    from pstats import SortKey

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()