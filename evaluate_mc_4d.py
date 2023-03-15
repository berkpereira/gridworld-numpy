import numpy as np
import dynamic_programming_4d as dp4
import monte_carlo_4d as mc4
import benchmark_problems_4d as bp4
import os
import matplotlib.pyplot as plt

def epsilon_train_policies(MDP, no_epsilon_params, no_episodes, no_steps):
    for epsilon in np.linspace(0.0, 1.0, no_epsilon_params):
        epsilon = round(epsilon, 2) # round to 2 decimal places
        # training MDP same as evaluation MDP except for direction_probability
        initial_policy = dp4.random_walk
        trained_policy, trained_policy_array = mc4.monte_carlo_policy_iteration(initial_policy,
                                                                                MDP, epsilon, no_episodes, no_steps)

        
        file_name = 'trained_array_epsilon_' + str(epsilon)
        file_name = file_name.replace('.', ',') # get rid of dots in file name
        np.save('results/4d/training_epsilon/' + file_name, trained_policy_array)

def epsilon_evaluate_policies(MDP, no_evaluations, epsilon_params):
    # for each evaluation wind parameter, we have a column vector of the average scores of policies TRAINED with different wind parameters
    MDP = dp4.MarkovGridWorld(MDP.grid_size, 1, MDP.direction_probability, MDP.obstacles, MDP.landing_zone, MDP.max_altitude)
    
    no_epsilon_params = len(epsilon_params)
    evaluations = np.zeros(shape=no_epsilon_params)
    to_go = no_epsilon_params
    i = 0
    for epsilon in epsilon_params:
        epsilon = round(epsilon, 2)

        # fetch the corresponding pre-computed policy
        file_name = 'results/4d/training_epsilon/trained_array_epsilon_' + str(epsilon)
        file_name = file_name.replace('.', ',')
        file_name = file_name + '.npy'
        policy_array = np.load(file_name)
        policy = dp4.array_to_policy(policy_array, MDP)

        cumulative_score = 0
        # run {no_evaluations} simulations using the fetched policy and record returns
        for _ in range(no_evaluations):
            history = mc4.generate_episode(MDP, policy)
            cumulative_score += history[-1,-1]
        
        average_score = cumulative_score / no_evaluations
        evaluations[i] = average_score    
        i += 1
        
        print(f'Another evaluation done. {to_go} more to go!')
        to_go -= 1
    return evaluations

if __name__ == "__main__":
    epsilon_train_policies(bp4.epsilon_MDP, 6, bp4.epsilon_no_episodes, bp4.epsilon_no_steps)