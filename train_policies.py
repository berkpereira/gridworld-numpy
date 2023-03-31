import dynamic_programming_4d as dp4
import monte_carlo_4d as mc4
import dynamic_programming_3d as dp3
import monte_carlo_3d as mc3
import numpy as np
import pickle

# define MDP file name parameters.
# upper limits are exclusive in these range functions
grid_sizes = range(4, 14)
IDs = range(1, 4)
wind_params = np.arange(0.70, 1.05, 0.05)

"""

DEFINE TRAINING HYPERPARAMETERS

"""
# 3D
wind_train_3d = 0.9      # DP
epsilon_train_3d = 0.1   # MC
ratio_episodes_3d = 0.25 # MC
no_policy_steps_3d = 3   # MC

# 4D
wind_train_4d = 0.9     # DP
epsilon_train_4d = 0.2  # MC
ratio_episodes_4d = 0.3 # MC
no_policy_steps_4d = 3  # MC


# TO SWITCH BETWEEN 3D and 4D, CHANGE THE INT BELOW
dimension = 4


for grid_size in grid_sizes:
    for ID in IDs:
        if dimension == 3:
            MDP_file = f"benchmark-problems/3d/{grid_size}{ID}_wind_{str(wind_train_3d).replace('.', ',')}.p"
        elif dimension == 4:
            MDP_file = f"benchmark-problems/4d/{grid_size}{ID}_wind_{str(wind_train_4d).replace('.', ',')}.p"
        else:
            Exception("Invalid dimension! Must be either 3 or 4.")
        
        with open(MDP_file, 'rb') as f:
            train_MDP = pickle.load(f) # load MDP for training (with correct wind parameter as decided for training)
        

        if dimension == 3:
            # train Dynamic Programming policy
            dp_policy, dp_policy_array, dp_train_time = dp3.value_iteration(dp3.random_walk, train_MDP, np.inf, train_time = True)
            
            # train Monte Carlo policy
            no_episodes_3d = int(np.ceil(ratio_episodes_3d * train_MDP.state_space.shape[0]))
            mc_policy, mc_policy_array, mc_train_time = mc3.monte_carlo_policy_iteration(dp3.random_walk, train_MDP, epsilon_train_3d, no_episodes_3d, no_policy_steps_3d)
        elif dimension == 4:
            # train Dynamic Programming policy

            
            # train Monte Carlo policy
            no_episodes_4d = int(np.ceil(ratio_episodes_4d * train_MDP.state_space.shape[0]))
            mc_policy, mc_policy_array = mc4.monte_carlo_policy_iteration(dp4.random_walk, train_MDP, epsilon_train_4d, no_episodes_4d, no_policy_steps_4d)
        else:
            Exception("Invalid dimension! Must be either 3 or 4.")