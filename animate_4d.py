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


if __name__ == "__main__":
    grid_size = 10
    ID = 3
    wind = 1
    method = 'dp'

    # remember state format: altitude, heading, x, y
    initial_state = np.array([grid_size * 2, 1, 8, 2])

    if wind == 1:
        MDP_file = f'benchmark-problems/4d/{grid_size}{ID}_wind_1,0.p'
    else:
        MDP_file = f'benchmark-problems/4d/{grid_size}{ID}_wind_{str(round(wind, 2)).replace(".", ",")}.p' 
    with open(MDP_file, 'rb') as f:
        MDP = pickle.load(f)
    
    policy_file = f'benchmark-policies/4d/{method}/{grid_size}{ID}_wind_0,9_policy_array.npy'
    policy = dp4.array_to_policy(np.load(policy_file), MDP)
    history = mc4.generate_episode(MDP, policy, initial_state=initial_state)
    mc4.play_episode(MDP, policy, history, save = True, file_name=f'{fsp.fig_path}/4d-{method}-trajectory.pdf')
    plt.close()


    # Now find MIP solution to the same problem
    ampl = mip4.initialise_ampl()
    mip_history, actions, solve_time = mip4.mip_history_and_actions_from_mdp(MDP, initial_state[2:4], initial_state[1], ampl)
    mc4.play_episode(MDP, None, mip_history, save = True, file_name = f'{fsp.fig_path}/4d-ip-trajectory.pdf')
