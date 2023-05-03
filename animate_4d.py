# this file serves as a "playground" for miscellaneous animating of agent policies and MIP solutions

import monte_carlo_4d as mc4
import dynamic_programming_4d as dp4
import solve_mip4d as mip4
import evaluate_mip_4d as emip4
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

def convert_params_to_MDP(grid_size, ID, wind):
    if wind == 1:
        MDP_file = f'benchmark-problems/4d/{grid_size}{ID}_wind_1,0.p'
    else:
        MDP_file = f'benchmark-problems/4d/{grid_size}{ID}_wind_{str(round(wind, 2)).replace(".", ",")}.p'
    
    with open(MDP_file, 'rb') as f:
        MDP = pickle.load(f)
    return MDP

def example_trajectory_3d(method_str, grid_size, ID, wind, initial_2d_position, initial_velocity, save=False, two_d=False, width='half', el_az_array = None):
    # remember state format: altitude, heading, x, y
    initial_state = np.zeros(shape=4, dtype='int32')
    initial_state[0] = grid_size * 2
    initial_state[1] = initial_velocity
    initial_state[2] = initial_2d_position[0]
    initial_state[3] = initial_2d_position[1]

    MDP = convert_params_to_MDP(grid_size, ID, wind)
    
    if method_str in ['MC', 'DP']:
        policy_file = f'benchmark-policies/4d/{method_str.lower()}/{grid_size}{ID}_wind_0,9_policy_array.npy'
        policy = dp4.array_to_policy(np.load(policy_file), MDP)
        history = mc4.generate_episode(MDP, policy, initial_state=initial_state)
        mc4.play_episode(MDP, policy, history, save=save, two_d=two_d, file_name=f'{fsp.fig_path}/4d-{method_str}-allowable-trajectory.pdf', width=width, el_az_array=el_az_array, method_str=method_str)
    elif method_str == 'IP':
        history, _, solve_time = emip4.mip_simulate_closed_loop(MDP, initial_state[2:4], initial_state[1])
        print(f'MILP cumulative solve time: {solve_time} seconds')
        mc4.play_episode(MDP, None, history, save=save, two_d=two_d, file_name=f'{fsp.fig_path}/4d-{method_str}-allowable-trajectory.pdf', width=width, el_az_array=el_az_array, method_str=method_str)

if __name__ == "__main__":
    # INPUT DATA
    method_str = 'MC'
    grid_size = 13
    ID = 2
    wind = 0.8
    no_episodes = 5
    
    # PRESCRIBED INITIAL CONDIITONS?
    prescribed_start = False

    # IF USING PRESCRIBED INITIAL CONDITIONS
    if prescribed_start:
        initial_2d_position = np.array([5, 2], dtype='int32')
        initial_velocity = 2


    if not prescribed_start:
        MDP = convert_params_to_MDP(grid_size, ID, wind)
        if method_str in ['MC', 'DP']:
            policy_file = f'benchmark-policies/4d/{method_str.lower()}/{grid_size}{ID}_wind_0,9_policy_array.npy'
            policy = dp4.array_to_policy(np.load(policy_file), MDP)
            for _ in range(no_episodes):
                history = mc4.generate_episode(MDP, policy)
                mc4.play_episode(MDP, policy, history, method_str=method_str)
        else:
            print('Invalid method string. Recall we do NOT have a valid license for CPLEX anymore!')
    else:
        # IF WE WANT TO CHOOSE THE INITIAL CONDITIONS
        example_trajectory_3d(method_str, grid_size, ID, wind, initial_2d_position, initial_velocity)