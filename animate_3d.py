# this file serves as a "playground" for miscellaneous animating of agent policies and MIP solutions

import monte_carlo_3d as mc3
import dynamic_programming_3d as dp3
import solve_mip3d as mip3
import numpy as np
import copy


if __name__ == "__main__":
    import benchmark_problems_3d as bp3
    #policy_name = 'trained_array_wind_0,8'
    #file_name = 'results/4d/training_wind/' + policy_name
    policy_name = 'trained_array_wind_1,0'
    file_name = 'results/3d/training_wind/' + policy_name
    file_name = file_name.replace('.', ',')
    file_name = file_name + '.npy'
    policy_array = np.load(file_name)

    #policy = dp4.array_to_policy(policy_array, bp4.wind_MDP)
    policy = dp3.array_to_policy(policy_array, bp3.wind_MDP)
    
    MDP = copy.deepcopy(bp3.wind_MDP)
    MDP.direction_probability = 1
    MDP.prob_other_directions = (1 - MDP.direction_probability) / 4
    mc3.simulate_policy(MDP, policy, 10)
