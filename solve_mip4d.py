import os
import pandas as pd
import time
import math
import numpy as np
from monte_carlo_4d import *
from amplpy import AMPL, Environment

os.system('clear')
path_to_ampl_exec = "/Users/gabrielpereira/ampl.macos64"
path_to_this_repo = "/Users/gabrielpereira/repos/gridworld-numpy"

def solve_mip(ampl):
    # load ampl installation
    #ampl = AMPL(Environment(path_to_ampl_exec))
    #print(ampl.get_option('version')) # check version, etc.

    # read model and data files
    #ampl.read(path_to_this_repo + "/ampl4d/new-mip-4d.mod")
    #ampl.read_data(path_to_this_repo + "/ampl4d/mip-4d.dat")

    # specify the solver to use
    ampl.option["solver"] = "cplex"

    # solve the problem
    st = time.time()
    ampl.solve()
    et = time.time()
    print(f'Done in {et - st} seconds.')


    # stop here if the model was not solved
    assert ampl.get_value("solve_result") == "solved"

    # get cost function
    objective = ampl.get_objective('LandingError')
    print(f'Objective function (landing error): {objective.value()}')
    return ampl

# ampl object is the input here
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
    else:
        raise Exception("No valid direction conversion!")

# initial_pd input is the ampl entity after pandas conversion
# velocities input is the numpy matrix
def convert_to_history(velocities, initial_pd):
    
    # extra column would accommodate reward! just for compatibility with plotting functions originally for RL
    history = np.zeros(shape = (velocities.shape[0], velocities.shape[1] + 1))

    history[0,2:4] = np.array([initial_pd["initial"][0], initial_pd["initial"][1]])
    altitude = velocities.shape[0] - 1
    history[0,0] = altitude
    for i in range(1, velocities.shape[0]):
        altitude -= 1
        velocity_direction = binary_velocity_to_direction(velocities[i-1])
        history[i,2:4] = history[i-1,2:4] + velocity_direction
        history[i,1] = np.where(velocities[i-1] == 1)[0][0] # convert to representation of heading in (0,1,2,3)
        history[i,0] = altitude
    return history


# this functions takes as input an ampl objective with already read model and data files to begin with.
# Only then does it modifie the model data in accordance with the input MDP.
def mip_history_and_actions_from_mdp(MDP, initial_state, initial_velocity_index, ampl):
    mip_max_altitude = ampl.get_parameter('T')
    mip_max_altitude.set(MDP.max_altitude)

    mip_landing_zone = ampl.get_parameter('landing')
    mip_landing_zone.set_values([MDP.landing_zone[0], MDP.landing_zone[1]])

    mip_grid_size = ampl.get_parameter('grid_size')
    mip_grid_size.set(MDP.grid_size)

    mip_initial_state = ampl.get_parameter('initial')
    mip_initial_state.set_values([initial_state[0], initial_state[1]])

    mip_initial_velocity_index = ampl.get_parameter('initial_velocity_index')
    mip_initial_velocity_index.set(initial_velocity_index)

    # Here we will also address obstacles at some point.
    #
    #

    # solve integer optimisation problem
    ampl = solve_mip(ampl)

    # Fetch velocities and put them into a suitable data shape.
    #
    #
    #
    #
    velocities = ampl.get_variable('Velocity')
    velocities = velocities.get_values().to_pandas()
    velocities = velocities.to_numpy()
    velocities = get_velocities(velocities)

    # convert initial state and velocity information into a history matrix:
    initial_pd = ampl.get_parameter('initial').get_values().to_pandas()
    history = convert_to_history(velocities, initial_pd)

    return history

# this function expects velocites in some form that I must choose still. Perhaps a matrix of 2-element array velocities?
def actions_from_mip_variables(velocities):
    pass

if __name__ == "__main__":
    ampl = AMPL(Environment(path_to_ampl_exec))
    ampl.read(path_to_this_repo + "/ampl4d/new-mip-4d.mod")
    ampl.read_data(path_to_this_repo + "/ampl4d/mip-4d.dat")
    MDP = MarkovGridWorld(grid_size=5, obstacles=np.array([[]]), landing_zone = np.array([0,4]), max_altitude=11)
    initial_state = [2,1]
    
    # mip_history_from_mdp is the crucial function here 
    history = mip_history_and_actions_from_mdp(MDP, initial_state, 2, ampl)
    play_episode(MDP, None, history)