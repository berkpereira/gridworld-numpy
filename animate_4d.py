# this file serves as a "playground" for miscellaneous animating of agent policies and MIP solutions

import monte_carlo_4d as mc4
import dynamic_programming_4d as dp4
import solve_mip4d as mip4
import numpy as np
import copy


if __name__ == "__main__":
    import benchmark_problems_4d as bp4
    policy_name = 'trained_array_wind_1,0'
    file_name = 'results/4d/training_wind/' + 'trained_array_wind_1,0'
    file_name = file_name.replace('.', ',')
    file_name = file_name + '.npy'
    policy_array = np.load(file_name)

    policy = dp4.array_to_policy(policy_array, bp4.wind_MDP)
    
    
    MDP = copy.deepcopy(bp4.wind_MDP)
    MDP.direction_probability = 0.8
    MDP.prob_other_directions = (1 - MDP.direction_probability) / 2
    mc4.simulate_policy(MDP, policy, 10, policy_name)