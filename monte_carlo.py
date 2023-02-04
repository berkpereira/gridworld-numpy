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
    history = np.zeros(shape=(MDP.max_altitude + 2, no_state_dimensions + 1))

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
            if np.array_equal(history[row - 1][:3], np.array([1,0,0])):
                return history[:row+1]
            else:
                return history[:row]


def sample_policy(MDP, policy, state):
    rng = np.random.default_rng()
    stochastics = np.zeros(len(MDP.action_space))
    for action in MDP.action_space:
        stochastics[action] = policy(action, state)
    sampled_action = rng.choice(len(MDP.action_space), p=stochastics)
    return sampled_action

def play_episode(MDP, history):
    fig = plt.figure(figsize=(12,9))
    ax = fig.add_subplot(projection="3d")
    ax.set_aspect('equal')
    ax.grid()
    marker_size = 400

    def animate(i):
        if i == 0:
            ax.clear()
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
            for obstacle in MDP.obstacles:
                no_points = 50
                x_obstacle = np.full((no_points, 1), obstacle[0])
                y_obstacle = np.full((no_points, 1), obstacle[1])
                z_obstacle = np.linspace(0, MDP.max_altitude, no_points)

                ax.scatter(x_obstacle, y_obstacle, z_obstacle, marker="x", c='black', s=marker_size*2, alpha=0.3)

        if history[i][0] == 0:
            return ax.scatter(history[i][1], history[i][2], 0, marker="x", c='red', s=marker_size, alpha=1),
        for obstacle in MDP.obstacles:
            if np.array_equal(history[i][1:3], obstacle):
                return ax.scatter(history[i][1], history[i][2], history[i][0], marker="x", c='red', s=marker_size, alpha=1),
        if history[i][0] > MDP.max_altitude:
            return ax.scatter(history[i][1], history[i][2], 0, marker="h", c='green', s=marker_size, alpha=1),
        return ax.scatter(history[i][1], history[i][2], history[i][0], marker="P", c='blue', s=marker_size, alpha=1),


    ani = animation.FuncAnimation(plt.gcf(), animate, frames=range(history.shape[0]), interval=500, repeat=False)
    plt.show()

def simulate_policy(MDP, policy, no_episodes=5):
    print(f'Generating episodes with policy {policy}')
    input('Press Enter to continue...')
    print()
    for i in range(no_episodes):
        history = generate_episode(MDP, policy)
        print(f'Episode number {i}')
        print('Episode history (state, then reward):')
        print(history)
        print()
        play_episode(MDP, history)

def run_random_then_optimal(MDP, policy, no_episodes):
    os.system('clear')
    simulate_policy(MDP, policy, no_episodes)
    print()
    print()
    print()
    print()
    print('Now running value iteration to converge on optimal policy!')
    input('Press Enter to continue...')
    optimal_policy, optimal_policy_array = value_iteration(policy, MDP, 20)
    simulate_policy(MDP, optimal_policy, no_episodes)



if __name__ == '__main__':
    MDP = MarkovGridWorld(grid_size = 5, direction_probability=1)
    run_random_then_optimal(MDP, random_walk, no_episodes=5)
