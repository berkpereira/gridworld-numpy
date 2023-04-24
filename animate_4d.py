# this file serves as a "playground" for miscellaneous animating of agent policies and MIP solutions

import monte_carlo_4d as mc4
import dynamic_programming_4d as dp4
import solve_mip4d as mip4
import numpy as np
import copy
import os
import pickle
import matplotlib.pyplot as plt
import fig_specs as fsp

# plot trajectories using different methods and as seen from above. For report.
def see_trajectories_2d(save=False):
    grid_size = 8
    ID = 3
    wind = 1
    method = 'mc'
    # remember state format: altitude, heading, x, y
    initial_state = np.array([grid_size * 2, 2, 4, 4])


    if wind == 1:
        MDP_file = f'benchmark-problems/4d/{grid_size}{ID}_wind_1,0.p'
    else:
        MDP_file = f'benchmark-problems/4d/{grid_size}{ID}_wind_{str(round(wind, 2)).replace(".", ",")}.p' 
    with open(MDP_file, 'rb') as f:
        MDP = pickle.load(f)
    
    policy_file = f'benchmark-policies/4d/{method}/{grid_size}{ID}_wind_0,9_policy_array.npy'
    policy = dp4.array_to_policy(np.load(policy_file), MDP)
    history = mc4.generate_episode(MDP, policy, initial_state=initial_state)
    mc4.play_episode(MDP, policy, history, save=save, two_d = True, file_name=f'{fsp.fig_path}/4d-{method}-trajectory.pdf')
    plt.close()


    # Now find MIP solution to the same problem
    ampl = mip4.initialise_ampl()
    mip_history, actions, solve_time = mip4.mip_history_and_actions_from_mdp(MDP, initial_state[2:4], initial_state[1], ampl)
    mc4.play_episode(MDP, None, mip_history, save=save, two_d = True, file_name = f'{fsp.fig_path}/4d-ip-trajectory.pdf')

def example_trajectory_3d(method_str, grid_size, ID, wind, initial_2d_position, initial_velocity, save=False, two_d=False, width='half', el_az_array = None):
    # remember state format: altitude, heading, x, y
    initial_state = np.zeros(shape=4, dtype='int32')
    initial_state[0] = grid_size * 2
    initial_state[1] = initial_velocity
    initial_state[2] = initial_2d_position[0]
    initial_state[3] = initial_2d_position[1]


    if wind == 1:
        MDP_file = f'benchmark-problems/4d/{grid_size}{ID}_wind_1,0.p'
    else:
        MDP_file = f'benchmark-problems/4d/{grid_size}{ID}_wind_{str(round(wind, 2)).replace(".", ",")}.p'
    with open(MDP_file, 'rb') as f:
        MDP = pickle.load(f)
    
    policy_file = f'benchmark-policies/4d/{method_str.lower()}/{grid_size}{ID}_wind_0,9_policy_array.npy'
    policy = dp4.array_to_policy(np.load(policy_file), MDP)
    history = mc4.generate_episode(MDP, policy, initial_state=initial_state)
    mc4.play_episode(MDP, policy, history, save=save, two_d=two_d, file_name=f'{fsp.fig_path}/4d-{method_str}-allowable-trajectory.pdf', width=width, el_az_array=el_az_array, method_str=method_str)



if __name__ == "__main__":
    save = False

    method = 'MC'
    grid_size = 8
    ID = 2
    wind = 0.9
    initial_2d_position = np.array([1, 4], dtype='int32')
    initial_velocity = 0

    el_az_array = [45,45]

    example_trajectory_3d(method, grid_size, ID, wind, initial_2d_position, initial_velocity, save = save, el_az_array = el_az_array)