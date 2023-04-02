import os
import pandas as pd
import time
import math
import numpy as np
from monte_carlo_3d import *
from amplpy import AMPL, Environment

path_to_ampl_exec = "/Users/gabrielpereira/ampl.macos64"
path_to_this_repo = "/Users/gabrielpereira/repos/gridworld-numpy"

def initialise_ampl():
    path_to_ampl_exec = "/Users/gabrielpereira/ampl.macos64"
    path_to_this_repo = "/Users/gabrielpereira/repos/gridworld-numpy"
    ampl = AMPL(Environment(path_to_ampl_exec))
    ampl.read(path_to_this_repo + "/ampl3d/new-mip-3d.mod")
    ampl.read_data(path_to_this_repo + "/ampl3d/mip-3d.dat")
    return ampl

def solve_mip(ampl):
    # load ampl installation
    #ampl = AMPL(Environment(path_to_ampl_exec))
    #print(ampl.get_option('version')) # check version, etc.

    # read model and data files
    #ampl.read(path_to_this_repo + "/ampl3d/mip-3d.mod")
    #ampl.read_data(path_to_this_repo + "/ampl3d/mip-3d.dat")

    # specify the solver to use
    ampl.option["solver"] = "cplex"

    # solve the problem
    st = time.time()
    ampl.solve()
    et = time.time()
    solve_time = et - st
    print(f'Done in {solve_time} seconds.')


    # stop here if the model was not solved
    try:
        assert ampl.get_value("solve_result") == "solved"
    except:
        return False, False

    # get landing error (cost function)
    objective = ampl.get_objective('LandingError')
    print(f'Objective function (landing error): {objective.value()}')
    return ampl, solve_time

# ampl object is the input here
# NO LONGER USED
def mdp_from_mip(ampl):
    max_altitude = int(ampl.get_parameter('T').value())
    grid_size = int(ampl.get_parameter('grid_size').value())
    landing_zone = np.array([ampl.get_parameter('landing').get_values().to_pandas()['landing'][0], ampl.get_parameter('landing').get_values().to_pandas()['landing'][1]], dtype='int32')
    
    # missing OBSTACLES
    MDP = MarkovGridWorld(grid_size=grid_size, obstacles =np.array([], dtype='int32') , landing_zone=landing_zone, max_altitude=max_altitude)
    return MDP

def get_velocities(velocities_vec):
    reshaped = np.zeros(shape=(int(velocities_vec.shape[0] / 4), 4))
    for i in range(reshaped.shape[0]):
        reshaped[i,:] = (velocities_vec[4*i:(4*i) + 4].flatten())
    return reshaped

def binary_velocity_to_direction(binary_decision_vec):
    if math.isclose(binary_decision_vec[0], 1): # down
        return np.array([1,0], dtype='int32')
    elif math.isclose(binary_decision_vec[1], 1): # right
        return np.array([0,1], dtype='int32')
    elif math.isclose(binary_decision_vec[2], 1): # up
        return np.array([-1,0], dtype='int32')
    elif math.isclose(binary_decision_vec[3], 1): # left
        return np.array([0,-1], dtype='int32')
    else: # stay put
        #raise Exception("No valid direction conversion!")
        return np.array([0,0], dtype='int32')

# initial_pd input is the ampl entity after pandas conversion
# velocities input is the numpy matrix
def convert_to_history(velocities, initial_pd):
    
    # extra column would accommodate reward! just for compatibility with plotting functions originally for RL
    history = np.zeros(shape = (velocities.shape[0], 4))

    history[0,1:3] = np.array([initial_pd["initial"][0], initial_pd["initial"][1]])
    altitude = velocities.shape[0] - 1
    history[0,0] = altitude
    for i in range(1, velocities.shape[0]):
        altitude -= 1
        velocity_direction = binary_velocity_to_direction(velocities[i-1])
        history[i,1:3] = history[i-1,1:3] + velocity_direction
        history[i,0] = altitude
    return history

# this functions takes as input an ampl objective with already read model and data files to begin with.
# Only then does it modifie the model data in accordance with the input MDP.    
def mip_history_and_actions_from_mdp(MDP, initial_state, ampl):
    mip_max_altitude = ampl.get_parameter('T')
    mip_max_altitude.set(MDP.max_altitude)

    mip_landing_zone = ampl.get_parameter('landing')
    mip_landing_zone.set_values([MDP.landing_zone[0], MDP.landing_zone[1]])

    mip_grid_size = ampl.get_parameter('grid_size')
    mip_grid_size.set(MDP.grid_size)

    mip_initial_state = ampl.get_parameter('initial')
    mip_initial_state.set_values([initial_state[0], initial_state[1]])

    # Here we will also address obstacles.
    # We drop the obstacles constraint if there aren't any.
    mip_no_obstacles = ampl.get_parameter('no_obstacles')
    mip_no_obstacles.set(MDP.obstacles.shape[0])
    mip_obstacles = ampl.get_parameter('obstacles')
    if MDP.obstacles.shape[1] > 0:
        for i in range(MDP.obstacles.shape[0]):
            mip_obstacles[i,0] = int(MDP.obstacles[i,0])
            mip_obstacles[i,1] = int(MDP.obstacles[i,1])
    else:
        obstacle_constraint_one = ampl.get_constraint('AvoidObstacleRowPos')
        obstacle_constraint_two = ampl.get_constraint('AvoidObstacleRowNeg')
        obstacle_constraint_three = ampl.get_constraint('AvoidObstacleColPos')
        obstacle_constraint_four = ampl.get_constraint('AvoidObstacleColNeg')
        obstacle_constraint_one.drop()
        obstacle_constraint_two.drop()
        obstacle_constraint_three.drop()
        obstacle_constraint_four.drop()


    # solve integer optimisation problem
    ampl, solve_time = solve_mip(ampl)

    # solution failed!
    if ampl is False: # return number of arguments equal to regular output! thus doesn't break unpacking when called 
        return False, False, False

    # Fetch velocities and put them into a suitable data shape.
    velocities = ampl.get_variable('Velocity')
    velocities = velocities.get_values().to_pandas()
    velocities = velocities.to_numpy()
    velocities = get_velocities(velocities)

    # convert initial state and velocity information into a history matrix:
    initial_pd = ampl.get_parameter('initial').get_values().to_pandas()
    history = convert_to_history(velocities, initial_pd)

    actions = actions_from_mip_variables(velocities, MDP.max_altitude)

    return history, actions, solve_time    

# this function expects velocites in a (T+1)x4 matrix of binary variables
def actions_from_mip_variables(velocities, max_altitude):
    actions = np.zeros(shape=max_altitude, dtype='int32')

    """
    # first action must always be 0. This is because the MIP formulation is limited in its first time step to the initial velocity,
    # whereas the DP formulation allows the agent to "override" the initial velocity using its first action.
    for i in range(1, max_altitude):
        if np.array_equal(velocities[i], velocities[i-1]):
            actions[i] = 0
        elif np.where(velocities[i] == 1)[0][0] == np.where(velocities[i-1] == 1)[0][0] + 1:
            actions[i] = 2
        elif np.where(velocities[i] == 1)[0][0] == np.where(velocities[i-1] == 1)[0][0] - 1:
            actions[i] = 1
        elif velocities[i,0] == 1: # edge case no. 1
            actions[i] = 2
        else: # edge case no. 2
            actions[i] = 1
    """
    
    # quite different for 3D state space case:
    for i in range(max_altitude):
        if velocities[i,0] == 1:
            actions[i] = 1
        elif velocities[i,1] == 1:
            actions[i] = 2
        elif velocities[i,2] == 1:
            actions[i] = 3
        elif velocities[i,3] == 1:
            actions[i] = 4
        else: # must be stay put
            actions[i] = 0
    return actions

if __name__ == "__main__":
    ampl = initialise_ampl()
    #MDP = MarkovGridWorld(grid_size=20, obstacles=np.array([[4,4], [5,6], [13,4], [3,16], [12,12], [16,6]]), landing_zone = np.array([6,6]), max_altitude=20)
    test_MDP = MarkovGridWorld(grid_size=5, direction_probability=1, obstacles=np.array([[0,0]]), landing_zone=np.array([2,2]), max_altitude=6)
    initial_state = [1,1]

    # mip_history_from_mdp is the crucial function here 
    history, actions, solve_time = mip_history_and_actions_from_mdp(test_MDP, initial_state, ampl)
    print(history)
    print()
    print(actions)
    print()
    print(f'Solve time: {solve_time} seconds')
    play_episode(test_MDP, None, history)