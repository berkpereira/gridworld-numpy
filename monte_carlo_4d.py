from dynamic_programming_4d import *
import numpy as np
import os


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.path as path
from scipy.spatial.distance import cityblock


def forward_policy(action, state):
    if action == 0:
        return 1
    else:
        return 0


# we will always begin episodes from max altitude, because we might as well and we thus cover
# more state visits per episode (no episodes where agent just crashes straight into the ground).
def generate_episode(MDP, policy):
    no_state_dimensions = MDP.state_space.shape[1] # number of columns = dimensions in state space

    # from nature of the problem, the max number of observed states in an episode (up to reaching MDP.termina_state)
    # is MDP.max_altitude + 2 (descend all of the altitude down to 1, then land, and then get taken to MDP.terminal_state).
    # Also note history is not an integer array, because the rewards observed are in general not integers.
    history = - np.ones(shape=(MDP.max_altitude + 2, no_state_dimensions + 1))

    # now we pick a state with altitude = MDP.max_altitude, with equal probability of any state.
    # this is initialising the state that then gets taken forward via sampling of problem dynamics and policy
    first_state_max_alt = np.where(MDP.state_space[:,0] == MDP.max_altitude)[0][0]
    
    # if we've accidentally picked an obstacle state to begin with, pick again until that's not the case.
    picked_obstacle = True
    while picked_obstacle is True:
        current_state = MDP.state_space[np.random.choice(np.arange(first_state_max_alt, first_state_max_alt + 4 * MDP.grid_size**2))]
        
        picked_obstacle = False
        for obstacle in MDP.obstacles:
            if np.array_equal(current_state[2:], obstacle):
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
        if np.array_equal(history[row][:4], MDP.terminal_state):
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
    marker_size = 400

    taper_offset = 0.3
    semi_span = 1.3
    marker_vertices = np.array([[0.5,0], [0.5,0.3], [0.1,0.3], [0.1, 1], [semi_span, 1 - taper_offset], [semi_span,1.2 - taper_offset], [0.1, 1.2], [0.05, 1.6], [-0.05, 1.6], [-0.1, 1.2], [-semi_span, 1.2 - taper_offset], [-semi_span, 1 - taper_offset], [-0.1, 1], [-0.1, 0.3], [-0.5, 0.3], [-0.5, 0]])
    aircraft_marker = path.Path(vertices=marker_vertices)

    def animate(i):
        if i == 0:
            ax.clear()
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
            if np.array_equal(history[i][2:4], obstacle):
                ax.plot(history[:,2],history[:,3],history[:,0], 'r-.') # trajectory
                return ax.scatter(history[i][2], history[i][3], history[i][0], marker="x", c='red', s=marker_size, alpha=1),
        
        # landed, not obstacle because already checked.
        if history[i][0] == 0:
            normalised_manhattan = cityblock(history[i][2:4], MDP.landing_zone) / ((MDP.grid_size - 1) * 2)
            ax.plot(history[:,2],history[:,3],history[:,0], '-.', color=plt.cm.winter(1 - normalised_manhattan)) # trajectory
            return ax.scatter(history[i][2], history[i][3], 0, marker=aircraft_marker, color=plt.cm.winter(1 - normalised_manhattan), s=marker_size*1.5, alpha=1),
        return ax.scatter(history[i][2], history[i][3], history[i][0], marker=aircraft_marker,  c='brown', s=marker_size*1.5, alpha=1),



    ani = animation.FuncAnimation(plt.gcf(), animate, frames=range(history.shape[0]), interval=300, repeat=False)
    plt.show()

def simulate_policy(MDP, policy, no_episodes=5):
    print(f'Generating episodes with policy: {policy.__name__}')
    #os.system(f'say "Generating episodes with policy: {policy.__name__}"')
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
    #os.system('say "Now running value iteration to converge on an optimal policy!"')
    input('Press Enter to continue...')
    optimal_policy, optimal_policy_array = policy_iteration(policy, MDP, 10, 1000)
    optimal_policy.__name__ = 'optimal_policy'
    simulate_policy(MDP, optimal_policy, no_episodes)

# first-visit Monte Carlo.
# this implementation ASSUMES DISCOUNT FACTOR = 1
def monte_carlo_policy_evaluation(MDP, policy, max_iterations):
    # we will use this array to count how many visits each state has had, so that we can then average the returns per state visit
    state_visits = np.zeros(shape=MDP.problem_shape)
    # we will use this array to accumulate the return observed
    cumulative_return = np.zeros(shape=MDP.problem_shape)

    for _ in range(max_iterations):
        episode = generate_episode(MDP, policy)
        
        # ASSUMING A DISCOUNT FACTOR OF 1, WE CAN SAY:
        observed_return = episode[-1][-1]
        
        for step in episode:
            state = step[:-1].astype('int32') # discard the last column, which is the reward
            state_visits[tuple(state)] += 1
            cumulative_return[tuple(state)] += observed_return
    
    value_estimate = np.divide(cumulative_return, state_visits, out=np.zeros_like(cumulative_return), where=state_visits!=0)
    return value_estimate


if __name__ == '__main__':
    os.system('clear')
    buildings = np.array([[]], ndmin=2, dtype='int32')
    MDP = MarkovGridWorld(grid_size = 3, max_altitude=3, obstacles = buildings, landing_zone = np.array([0,0], dtype='int32'), direction_probability=1)
    value_estimate = monte_carlo_policy_evaluation(MDP, random_walk, 50000)
    value_real = policy_evaluation(random_walk, MDP, initial_value=np.zeros(shape=MDP.problem_shape), epsilon=0, max_iterations=100000)
    print(value_estimate)
    print()
    print()
    print()
    print()
    print(value_real)
    print()
    print()
    print()
    print()
    print('Difference:')
    print(np.subtract(value_estimate, value_real))