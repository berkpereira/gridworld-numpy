from dynamic_programming_3d import *
import numpy as np
import os


import matplotlib.pyplot as plt
import matplotlib.animation as animation


# we will always begin episodes from max altitude, because we might as well and we thus cover
# more state visits per episode (no episodes where agent just crashes straight into the ground).
def generate_episode(MDP, policy):
    no_state_dimensions = MDP.state_space.shape[1] # number of columns = dimensions in state space

    # from nature of the problem, the max number of observed states in an episode (up to reaching MDP.termina_state)
    # is MDP.max_altitude + 2 (descend all of the altitude down to 1, then land, and then get taken to MDP.terminal_state).
    # Also note history is not an integer array, because the rewards observed are in general not integers.
    history = - np.ones(shape=(MDP.max_altitude + 2, no_state_dimensions + 1)) # initialise to terminal states

    # now we pick a state with altitude = MDP.max_altitude, with equal probability of any state.
    # this is initialising the state that then gets taken forward via sampling of problem dynamics and policy
    first_state_max_alt = np.where(MDP.state_space[:,0] == MDP.max_altitude)[0][0]
    
    # if we've accidentally picked an obstacle state to begin with, pick again until that's not the case.
    picked_obstacle = True
    while picked_obstacle is True:
        current_state = MDP.state_space[np.random.choice(np.arange(first_state_max_alt, first_state_max_alt + MDP.grid_size**2))]
        
        picked_obstacle = False
        for obstacle in MDP.obstacles:
            if np.array_equal(current_state[1:], obstacle):
                picked_obstacle = True
                break
                

    
    # initialise index which we will use to fill the rows of the history matrix
    time_level = 0

    # episode is over when current_state == MDP.terminal_state. That's the entire point of having an explicit terminal state.
    while not np.array_equal(current_state, MDP.terminal_state):
        history[time_level][:no_state_dimensions] = current_state
        history[time_level][-1] = MDP.reward(current_state)

        # take action to be a sample from an appropriate probability distribution over the action space, as
        # per the policy in use.
        action = sample_policy(MDP, policy, current_state)
        current_state = MDP.state_transition(current_state, action)
        time_level += 1
    
    history = truncate_terminal(MDP, history)
    return history

def truncate_terminal(MDP, history):
    for row in range(history.shape[0]):
        if np.array_equal(history[row][:3], MDP.terminal_state):
            return history[:row]


def sample_policy(MDP, policy, state):
    rng = np.random.default_rng()
    stochastics = np.zeros(len(MDP.action_space))
    for action in MDP.action_space:
        stochastics[action] = policy(action, state)
    sampled_action = rng.choice(len(MDP.action_space), p=stochastics)
    return sampled_action

def play_episode(MDP, policy, history):
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect('equal')
    ax.grid()
    marker_size = 200

    def animate(i):
        if i == 0:
            ax.clear()
            if policy is None:
                ax.set_title(f'Agent simulation from MIP solution.\nLanding zone (x,y): {tuple(MDP.landing_zone)}')
            else:    
                ax.set_title(f'Agent simulation under policy: {policy.__name__}\nDirection probability: {MDP.direction_probability}\nLanding zone (x,y): {tuple(MDP.landing_zone)}\nTotal return: {history[-1,-1]}')
            ax.axes.set_xlim3d(left=0, right=MDP.grid_size - 1)
            ax.axes.set_ylim3d(bottom=0, top=MDP.grid_size - 1)
            ax.axes.set_zlim3d(bottom=0, top=MDP.max_altitude)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            plt.xticks(np.arange(MDP.grid_size))
            plt.yticks(np.arange(MDP.grid_size))
            ax.set_zticks(np.arange(MDP.max_altitude + 1))
            
            # plot obstacle as a sort of building up to MDP.max_altitude.
            # need to make this proper, just a crappy demo as it stands.
            if MDP.obstacles.size != 0:
                for obstacle in MDP.obstacles:
                    no_points = 50
                    x_obstacle = np.full((no_points, 1), obstacle[0])
                    y_obstacle = np.full((no_points, 1), obstacle[1])
                    z_obstacle = np.linspace(0, MDP.max_altitude, no_points)

                    ax.scatter(x_obstacle, y_obstacle, z_obstacle, marker="h", c='black', s=marker_size*2, alpha=0.1)
            
            # also visualise landing zone
            ax.scatter(MDP.landing_zone[0], MDP.landing_zone[1], 0, marker='o', c='purple', s=marker_size)

        for obstacle in MDP.obstacles:
            if np.array_equal(history[i][1:3], obstacle):
                ax.plot(history[:,1],history[:,2],history[:,0], 'r-.') # trajectory
                return ax.scatter(history[i][1], history[i][2], history[i][0], marker="x", c='red', s=marker_size, alpha=1),
        
        # landed, not obstacle because already checked
        if history[i][0] == 0:
            normalised_manhattan = cityblock(history[i][1:3], MDP.landing_zone) / ((MDP.grid_size - 1) * 2)
            ax.plot(history[:,1],history[:,2],history[:,0], '-.', color=plt.cm.winter(1 - normalised_manhattan)) # plot trajectory
            return ax.scatter(history[i][1], history[i][2], 0, marker='P', color=plt.cm.winter(1 - normalised_manhattan), s=marker_size, alpha=1),
            
        return ax.scatter(history[i][1], history[i][2], history[i][0], marker="P",  c='brown', s=marker_size, alpha=1),


    ani = animation.FuncAnimation(plt.gcf(), animate, frames=range(history.shape[0]), interval=100, repeat=False)
    plt.show()

def simulate_policy(MDP, policy, no_episodes=5):
    print(f'Generating episodes with policy: {policy.__name__}')
    input('Press Enter to continue...')
    print()
    for i in range(no_episodes):
        history = generate_episode(MDP, policy)
        print(f'Episode number {i}')
        print('Episode history (state, then reward):')
        print(history)
        print()
        play_episode(MDP, policy, history)

def run_random_then_optimal(MDP, policy, no_episodes):
    os.system('clear')
    simulate_policy(MDP, policy, no_episodes)
    print()
    print()
    print()
    print()
    print('Now running value iteration to converge on an optimal policy!')
    input('Press Enter to continue...')
    optimal_policy, optimal_policy_array = value_iteration(policy, MDP, 20)
    optimal_policy.__name__ = 'optimal_policy'
    simulate_policy(MDP, optimal_policy, no_episodes)


# first-visit Monte Carlo.
# this implementation ASSUMES DISCOUNT FACTOR = 1
def monte_carlo_policy_evaluation(MDP, policy, no_episodes):
    # we will use this array to count how many visits each state has had, so that we can then average the returns per state visit
    state_visits = np.zeros(shape=MDP.problem_shape)
    # we will use this array to accumulate the return observed
    cumulative_return = np.zeros(shape=MDP.problem_shape)

    for _ in range(no_episodes):
        episode = generate_episode(MDP, policy)
        
        # ASSUMING A DISCOUNT FACTOR OF 1, WE CAN SAY:
        observed_return = episode[-1][-1]
        
        for step in episode:
            state = step[:-1].astype('int32') # discard the last column, which is the reward
            state_visits[tuple(state)] += 1
            cumulative_return[tuple(state)] += observed_return
    
    value_estimate = np.divide(cumulative_return, state_visits, out=np.zeros_like(cumulative_return), where=state_visits!=0)
    return value_estimate

# generates exploratory policy
def monte_carlo_array_to_policy(policy_array, MDP, epsilon):
    # 4D array used
    # 2nd, 3rd, 4th indices correspond to dimensions of the state space.
    # 1st index corresponds to action number.
    state_action_probabilities = np.zeros(shape = (len(MDP.action_space), MDP.problem_shape[0], MDP.problem_shape[1], MDP.problem_shape[2]))
    for index in np.ndindex(MDP.problem_shape[0], MDP.problem_shape[1], MDP.problem_shape[2]):
        state_action_probabilities[:,index[0], index[1], index[2]] = np.full(len(MDP.action_space), fill_value=epsilon / len(MDP.action_space))
        
        # overwrite the previous for the greedy action.
        greedy_action = policy_array[index]
        state_action_probabilities[(greedy_action,) + index] = 1 - epsilon + (epsilon / len(MDP.action_space)) # exploratory epsilon-greedy policy: keep exploring
    
    # policy function itself just has to index the 3D array we've created, which contains all the policy-defining information
    def policy(action, state):
        return state_action_probabilities[(action,) + tuple(state)]
    
    # return policy function
    return policy

def monte_carlo_policy_iteration(policy, MDP, exploration_epsilon, evaluation_no_episodes=50, improvement_max_iterations=50):
    iteration_count = 1
    policy_is_stable = False
    current_policy = policy
    current_policy_array = np.ones(shape=MDP.problem_shape, dtype='int32') * -10 # initialise greedy policy array to a bogus instance
    while policy_is_stable is False and iteration_count <= improvement_max_iterations:
        # as per Sutton Barto 2nd, chapter 4.3, next iteration is better-converging if we
        # start with the previous value estimate, hence the assignment into initial_value
        print(f'Iteration number: {iteration_count}')
        print(f'Terminal state: {MDP.terminal_state}')
        print('Current greedy policy array (disregard in iteration no. 1):')
        print(current_policy_array)
        print()

        initial_value = monte_carlo_policy_evaluation(MDP, current_policy, no_episodes=evaluation_no_episodes)
        print('Previous policy evaluation:')
        print(initial_value[:-1])
        new_policy_array = greedy_policy_array(initial_value, MDP)
        
        if np.array_equal(new_policy_array, current_policy_array):
            policy_is_stable = True
            print('Policy has stabilised.')
            print()
            break # stop iterating

        current_policy_array = new_policy_array
        current_policy = monte_carlo_array_to_policy(new_policy_array, MDP, epsilon=exploration_epsilon)
        iteration_count += 1
        
    print('Final policy array:')
    print(current_policy_array[:-1])
    print()
    # in the end, best to return a DETERMINISTIC VERSION OF THE POLICY
    final_policy = array_to_policy(current_policy_array, MDP)
    return final_policy, current_policy_array


if __name__ == '__main__':
    """
    buildings = np.array([[1,0], [1,2], [3,3], [5,4]], dtype='int32')
    MDP = MarkovGridWorld(grid_size = 6, max_altitude=14, obstacles = buildings, landing_zone = np.array([2,2], dtype='int32'), direction_probability=0.90)
    run_random_then_optimal(MDP, random_walk, no_episodes=5)
    """
    os.system('clear')
    #buildings = np.array([[1,1], [3,2], [4,1]], ndmin=2, dtype='int32'))
    grid_size = 12
    buildings = np.random.randint(0, grid_size, (6,2))
    MDP = MarkovGridWorld(grid_size = grid_size, max_altitude=12, obstacles = buildings, landing_zone = np.array([2,2], dtype='int32'), direction_probability=1)
    no_episodes = int(np.floor(len(MDP.state_space) / 2))
    no_steps = int(np.floor(len(MDP.state_space) / 40))
    no_episodes = 1000
    no_steps = 40
    print(f'Monte Carlo, number of episodes per improvement: {no_episodes}')
    print(f'Monte Carlo, number of improvement steps: {no_steps}')
    print()
    simulate_policy(MDP, random_walk, 5)
    print(f'Monte Carlo, number of episodes per improvement: {no_episodes}')
    print(f'Monte Carlo, number of improvement steps: {no_steps}')
    input('Press Enter to continue...')
    new_policy, new_policy_array = monte_carlo_policy_iteration(random_walk, MDP, 0.1, no_episodes, no_steps)
    simulate_policy(MDP, new_policy, 10)