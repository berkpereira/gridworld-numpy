import os
import pandas as pd
import time
import math
import numpy as np
from monte_carlo_3d import *
from amplpy import AMPL, Environment

path_to_ampl_exec = "/Users/gabrielpereira/ampl.macos64"
path_to_this_repo = "/Users/gabrielpereira/repos/gridworld-numpy"

def solve_mip():
    # load ampl installation
    ampl = AMPL(Environment(path_to_ampl_exec))
    #print(ampl.get_option('version')) # check version, etc.

    # read model and data files
    ampl.read(path_to_this_repo + "/ampl3d/mip-3d.mod")
    ampl.read_data(path_to_this_repo + "/ampl3d/mip-3d.dat")

    # specify the solver to use
    ampl.option["solver"] = "cplex"

    # solve the problem
    st = time.time()
    ampl.solve()
    et = time.time()
    print(f'Done in {et - st} seconds.')


    # stop here if the model was not solved
    assert ampl.get_value("solve_result") == "solved"

    # get number of turns in solution (cost function)
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

def reshape_velocities(velocities_vec):
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

    
    
    

# solve and get optimal variables
ampl = solve_mip()
velocities = ampl.get_variable('Velocity')
velocities = velocities.get_values().to_pandas()
velocities = velocities.to_numpy()
velocities = reshape_velocities(velocities)

initial_pd = ampl.get_parameter('initial').get_values().to_pandas()

print('History:')
print(convert_to_history(velocities, initial_pd))

MDP = mdp_from_mip(ampl)

history = convert_to_history(velocities, initial_pd)
play_episode(MDP, None, history)