from dynamic_programming_4d import *
import numpy as np
import os


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.path as path
from scipy.spatial.distance import cityblock
import time

import fig_specs as fsp
from matplotlib.transforms import Bbox


def forward_policy(action, state):
    if action == 0:
        return 1
    else:
        return 0


# we will always begin episodes from max altitude, because we might as well and we thus cover
# more state visits per episode (no episodes where agent just crashes straight into the ground).
def generate_episode(MDP, policy, initial_state = None):
    no_state_dimensions = MDP.state_space.shape[1] # number of columns = dimensions in state space

    # from nature of the problem, the max number of observed states in an episode (up to reaching MDP.termina_state)
    # is MDP.max_altitude + 2 (descend all of the altitude down to 1, then land, and then get taken to MDP.terminal_state).
    # Also note history is not an integer array, because the rewards observed are in general not integers.
    history = - np.ones(shape=(MDP.max_altitude + 2, no_state_dimensions + 1))

    # now we pick a state with altitude = MDP.max_altitude, with equal probability of any state.
    # this is initialising the state that then gets taken forward via sampling of problem dynamics and policy
    first_state_max_alt = np.where(MDP.state_space[:,0] == MDP.max_altitude)[0][0]
    
    # if we've accidentally picked an obstacle state to begin with, pick again until that's not the case.
    if initial_state is None:
        picked_obstacle = True
        while picked_obstacle is True:
            current_state = MDP.state_space[np.random.choice(np.arange(first_state_max_alt, first_state_max_alt + 4 * MDP.grid_size**2))]
            
            picked_obstacle = False
            for obstacle in MDP.obstacles:
                if np.array_equal(current_state[2:], obstacle):
                    picked_obstacle = True
                    break
    else: # allows for prescription of the initial state
        current_state = initial_state
                
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
    if policy.__name__ == "epsilon_greedy_policy":
        pass
    sampled_action = rng.choice(len(MDP.action_space), p=stochastics)
    return sampled_action

def play_episode(MDP, policy, history, policy_name=None, save = False, two_d = False, file_name = None, width = 'half', el_az_array = None, method_str = None):
    
    if not save:
        plt.rcParams['font.size'] = 13
        fig = plt.figure(figsize=(20,9.8))
        ax = fig.add_subplot(projection="3d")
        ax.set_aspect('equal')
        ax.grid()
        marker_size = 3000 / MDP.grid_size
        if el_az_array is not None:
            ax.view_init(elev=el_az_array[0], azim=el_az_array[1])
    else: # for saving plots
        if width == 'half':
                fig_width = fsp.text_width * fsp.text_width_factor
        elif width == 'full':
                fig_width = fsp.text_width
        if not two_d:
            fig_height = 2.5
            fig = plt.figure(figsize=(fig_width, fig_height))
            ax = fig.add_subplot(projection="3d")
            ax.set_aspect('equal')
            ax.grid()
            marker_size = 1000 / (MDP.grid_size * 3)
        else:
            fig = plt.figure(figsize=(fig_width, 1.8))
            ax = fig.add_subplot()
            ax.set_aspect('equal')
            ax.grid()
            marker_size = 1000 / (MDP.grid_size * fig_width)

    # create aeroplane-shaped marker
    taper_offset = 0.3
    semi_span = 1.3
    marker_vertices = np.array([[0.5,0], [0.5,0.3], [0.1,0.3], [0.1, 1], [semi_span, 1 - taper_offset], [semi_span,1.2 - taper_offset], [0.1, 1.2], [0.05, 1.6], [-0.05, 1.6], [-0.1, 1.2], [-semi_span, 1.2 - taper_offset], [-semi_span, 1 - taper_offset], [-0.1, 1], [-0.1, 0.3], [-0.5, 0.3], [-0.5, 0]])
    aircraft_marker = path.Path(vertices=marker_vertices)

    if not save:
        title_font_size = 16

        def animate(i):
            if i == 0:
                ax.clear()
                if policy is None:
                    ax.set_title(f'Agent simulation using MILP.\nDirection probability: {round(MDP.direction_probability, 2)}\nLanding zone (x,y): {tuple(MDP.landing_zone)}\nTotal return: {history[-1,-1]}')
                else:
                    if method_str is None:
                        if policy_name is None:    
                            ax.set_title(f'Agent simulation under policy.\nDirection probability: {round(MDP.direction_probability, 2)}\nLanding zone (x,y): {tuple(MDP.landing_zone)}\nTotal return: {history[-1,-1]}')
                        else:
                            ax.set_title(f'Agent simulation under policy: {policy_name}\nDirection probability: {round(MDP.direction_probability, 2)}\nLanding zone (x,y): {tuple(MDP.landing_zone)}\nTotal return: {history[-1,-1]}')
                    else:
                        if policy_name is None:    
                            ax.set_title(f'Agent simulation under {method_str} policy.\nDirection probability: {round(MDP.direction_probability, 2)}\nLanding zone (x,y): {tuple(MDP.landing_zone)}\nTotal return: {history[-1,-1]}')
                            plt.title(f'Agent simulation under {method_str} policy.\nDirection probability: {round(MDP.direction_probability, 2)}\nLanding zone (x,y): {tuple(MDP.landing_zone)}\nTotal return: {history[-1,-1]}', fontsize=title_font_size)
                        else:
                            ax.set_title(f'Agent simulation under {method_str} policy: {policy_name}\nDirection probability: {round(MDP.direction_probability, 2)}\nLanding zone (x,y): {tuple(MDP.landing_zone)}\nTotal return: {history[-1,-1]}')
                ax.axes.set_xlim3d(left=0, right=MDP.grid_size - 1)
                ax.axes.set_ylim3d(bottom=0, top=MDP.grid_size - 1)
                ax.axes.set_zlim3d(bottom=0, top=MDP.max_altitude)
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel('z')
                if MDP.grid_size <= 10:
                    plt.xticks(np.arange(MDP.grid_size))
                    plt.yticks(np.arange(MDP.grid_size))
                else:
                    plt.xticks(np.arange(0, MDP.grid_size, np.ceil(MDP.grid_size/8)))
                    plt.yticks(np.arange(0, MDP.grid_size, np.ceil(MDP.grid_size/8)))
                if MDP.max_altitude <= 15:
                    ax.set_zticks(np.arange(MDP.max_altitude + 1))
                else:
                    ax.set_zticks(np.arange(0, MDP.max_altitude + 1, np.ceil(MDP.max_altitude/10)))

                # plot obstacle as a sort of building up to MDP.max_altitude.
                # need to make this proper, just a crappy demo as it stands.
                if MDP.obstacles.size != 0:
                    for obstacle in MDP.obstacles:
                        no_points = 30
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



        ani = animation.FuncAnimation(plt.gcf(), animate, frames=range(history.shape[0]), interval=150, repeat=False)
        plt.show()
    else: # want to save figure
        if not two_d:
            for i in range(history.shape[0]):
                if i == 0:
                    ax.clear()

                    ax.axes.set_xlim3d(left=0, right=MDP.grid_size - 1)
                    ax.axes.set_ylim3d(bottom=0, top=MDP.grid_size - 1)
                    ax.axes.set_zlim3d(bottom=0, top=MDP.max_altitude)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    ax.set_zlabel('z')
                    if MDP.grid_size <= 20:
                        plt.xticks(np.arange(MDP.grid_size))
                        plt.yticks(np.arange(MDP.grid_size))
                    else:
                        plt.xticks(np.arange(0, MDP.grid_size, np.ceil(MDP.grid_size/10)))
                        plt.yticks(np.arange(0, MDP.grid_size, np.ceil(MDP.grid_size/10)))
                    if MDP.max_altitude <= 8:
                        ax.set_zticks(np.arange(MDP.max_altitude + 1))
                    else:
                        ax.set_zticks(np.arange(0, MDP.max_altitude + 1, np.ceil(MDP.max_altitude/2)))

                    # plot obstacle as a sort of building up to MDP.max_altitude.
                    # need to make this proper, just a crappy demo as it stands.
                    if MDP.obstacles.size != 0:
                        for obstacle in MDP.obstacles:
                            no_points = 30
                            x_obstacle = np.full((no_points, 1), obstacle[0])
                            y_obstacle = np.full((no_points, 1), obstacle[1])
                            z_obstacle = np.linspace(0, MDP.max_altitude, no_points)

                            ax.scatter(x_obstacle, y_obstacle, z_obstacle, marker="h", c='black', s=marker_size, alpha=0.1)
                    
                    # also visualise landing zone
                    ax.scatter(MDP.landing_zone[0], MDP.landing_zone[1], 0, marker='o', c='purple', s=marker_size)

                continue_crashed = False
                for obstacle in MDP.obstacles:
                    if np.array_equal(history[i][2:4], obstacle):
                        ax.plot(history[:,2],history[:,3],history[:,0], 'r-.') # trajectory
                        ax.scatter(history[i][2], history[i][3], history[i][0], marker="x", c='red', s=marker_size, alpha=1),
                        continue_crashed = True
                    if continue_crashed:
                        continue
                
                # landed, not obstacle because already checked.
                if history[i][0] == 0:
                    normalised_manhattan = cityblock(history[i][2:4], MDP.landing_zone) / ((MDP.grid_size - 1) * 2)
                    ax.plot(history[:,2],history[:,3],history[:,0], '-.', color=plt.cm.winter(1 - normalised_manhattan)) # trajectory
                    ax.scatter(history[i][2], history[i][3], 0, marker=aircraft_marker, color=plt.cm.winter(1 - normalised_manhattan), s=marker_size*1.5, alpha=1),
                    continue
                
                ax.scatter(history[i][2], history[i][3], history[i][0], marker=aircraft_marker,  c='brown', s=marker_size*1.5, alpha=1),
            ax.view_init(elev=37, azim=-57)
        else: # 2D version of the plot
            for i in range(history.shape[0]):
                if i == 0:
                    ax.clear()

                    ax.axes.set_xlim(-0.5, MDP.grid_size - 1 + 0.5)
                    ax.axes.set_ylim(-0.5, MDP.grid_size - 1 + 0.5)
                    ax.set_xlabel('x')
                    ax.set_ylabel('y')
                    if MDP.grid_size <= 20:
                        plt.xticks(np.arange(MDP.grid_size))
                        plt.yticks(np.arange(MDP.grid_size))
                    else:
                        plt.xticks(np.arange(0, MDP.grid_size, np.ceil(MDP.grid_size/10)))
                        plt.yticks(np.arange(0, MDP.grid_size, np.ceil(MDP.grid_size/10)))

                    # plot obstacle as a sort of building up to MDP.max_altitude.
                    if MDP.obstacles.size != 0:
                        for obstacle in MDP.obstacles:
                            x_obstacle = np.full((1, 1), obstacle[0])
                            y_obstacle = np.full((1, 1), obstacle[1])

                            ax.scatter(x_obstacle, y_obstacle, marker="h", c='black', s=marker_size, alpha=0.9)
                    
                    # also visualise landing zone
                    ax.scatter(MDP.landing_zone[0], MDP.landing_zone[1], marker='o', c='purple', s=marker_size)

                continue_crashed = False
                for obstacle in MDP.obstacles:
                    if np.array_equal(history[i][2:4], obstacle):
                        ax.plot(history[:,2],history[:,3], 'r-.') # trajectory
                        ax.scatter(history[i][2], history[i][3], marker="x", c='red', s=marker_size, alpha=1),
                        continue_crashed = True
                    if continue_crashed:
                        continue
                
                # landed, not obstacle because already checked.
                if history[i][0] == 0:
                    normalised_manhattan = cityblock(history[i][2:4], MDP.landing_zone) / ((MDP.grid_size - 1) * 2)
                    ax.plot(history[:,2],history[:,3], '-', color=plt.cm.winter(1 - normalised_manhattan)) # trajectory
                    ax.scatter(history[i][2], history[i][3], marker=aircraft_marker, color=plt.cm.winter(1 - normalised_manhattan), s=marker_size*1.5, alpha=1),
                    continue
                
                if i == 0:
                    ax.scatter(history[i][2], history[i][3], marker=aircraft_marker,  c='magenta', s=marker_size*1.5, alpha=1),    
                else:
                    ax.scatter(history[i][2], history[i][3], marker=aircraft_marker,  c='brown', s=marker_size*1.5, alpha=1),
            plt.grid(True)

        #plt.subplots_adjust(bottom=0, top=1, left=0.1, right=1)  # adjust padding
        plt.savefig(file_name, bbox_inches = Bbox([[0, -1], [fsp.text_width, fig_height]])) # [[x0, y0], [x1, y1]]
        plt.show()


def simulate_policy(MDP, policy, no_episodes=5, policy_name=None):
    if policy_name is None:
        print(f'Generating episodes with policy: {policy.__name__}')
    else:
        print(f'Generating episodes with policy: {policy_name}')
    input('Press Enter to continue...')
    print()
    for i in range(no_episodes):
        history = generate_episode(MDP, policy)
        print(f'Episode number {i}')
        print('Episode history (state, then reward):')
        print(history)
        print()
        play_episode(MDP, policy, history, policy_name)

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
    # 5D array used
    # 2nd, 3rd, 4th and 5th indices correspond to dimensions of the state space.
    # 1st index corresponds to action number.
    state_action_probabilities = np.zeros(shape = (len(MDP.action_space), MDP.problem_shape[0], MDP.problem_shape[1], MDP.problem_shape[2], MDP.problem_shape[3]))
    for index in np.ndindex(MDP.problem_shape[0], MDP.problem_shape[1], MDP.problem_shape[2], MDP.problem_shape[3]):
        state_action_probabilities[:,index[0], index[1], index[2], index[3]] = np.full(len(MDP.action_space), fill_value=epsilon / len(MDP.action_space))
        
        # overwrite the previous for the greedy action.
        greedy_action = policy_array[index]
        state_action_probabilities[(greedy_action,) + index] = 1 - epsilon + (epsilon / len(MDP.action_space)) # exploratory epsilon-greedy policy: keep exploring
    
    # policy function itself just has to index the 3D array we've created, which contains all the policy-defining information
    def policy(action, state):
        return state_action_probabilities[(action,) + tuple(state)]
    
    # return policy function
    return policy

def monte_carlo_policy_iteration(policy, MDP, exploration_epsilon, evaluation_no_episodes=50, improvement_max_iterations=50, train_time = False):
    iteration_count = 1
    policy_is_stable = False
    current_policy = policy
    current_policy_array = np.ones(shape=MDP.problem_shape, dtype='int32') * -10 # initialise greedy policy array to a bogus instance

    st = time.time()
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
    
    et = time.time()
    print('Final policy array:')
    print(current_policy_array[:-1])
    print()
    # in the end, best to return a DETERMINISTIC VERSION OF THE POLICY
    final_policy = array_to_policy(current_policy_array, MDP)

    if train_time is False:
        return final_policy, current_policy_array
    else:
        training_time = et - st
        return final_policy, current_policy_array, training_time


if __name__ == '__main__':
    os.system('clear')
    import benchmark_problems_4d as bp4

    MDP = bp4.epsilon_MDP
    no_episodes_ratio = 1
    no_episodes = int(np.ceil(no_episodes_ratio * MDP.state_space.shape[0]))
    no_steps = 5

    print(f'Monte Carlo, number of episodes per improvement: {no_episodes}')
    print(f'Monte Carlo, number of improvement steps: {no_steps}')
    input('Press Enter to continue...')
    new_policy, new_policy_array = monte_carlo_policy_iteration(random_walk, MDP, bp4.ratio_episodes_steps_epsilon, no_episodes, no_steps)
    
    simulate_policy(bp4.epsilon_MDP, new_policy, 10)
    