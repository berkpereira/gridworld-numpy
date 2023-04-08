# this file serves as a "playground" for miscellaneous animating of agent policies and MIP solutions

import monte_carlo_4d as mc4
import dynamic_programming_4d as dp4
import solve_mip4d as mip4
import numpy as np
import copy
import os
import pickle


if __name__ == "__main__":
    grid_size = 10
    ID = 2
    wind = 0.8
    method = 'dp'

    MDP_file = f'benchmark-problems/4d/{grid_size}{ID}_wind_{str(round(wind, 2)).replace(".", ",")}.p' 
    with open(MDP_file, 'rb') as f:
        MDP = pickle.load(f)
    
    policy_file = f'benchmark-policies/4d/{method}/{grid_size}{ID}_wind_0,9_policy_array.npy'
    policy = dp4.array_to_policy(np.load(policy_file), MDP)
    history = mc4.generate_episode(MDP, policy)
    mc4.play_episode(MDP, policy, history, save = False)
