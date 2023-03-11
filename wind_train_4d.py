import numpy as np
import dynamic_programming_4d as dp4
import monte_carlo_4d as mc4

def train_policies(evaluation_MDP, no_wind_parameters):
    for wind in np.linspace(0.0, 1.0, no_wind_parameters): # default is 50 different points
        wind = round(wind, 2) # round to 2 decimal places
        # training MDP same as evaluation MDP except for direction_probability
        training_MDP = dp4.MarkovGridWorld(grid_size=evaluation_MDP.grid_size, direction_probability=wind, obstacles=evaluation_MDP.obstacles, landing_zone=evaluation_MDP.landing_zone, max_altitude=evaluation_MDP.max_altitude)

        initial_policy = dp4.random_walk
        trained_policy, trained_policy_array = dp4.value_iteration(policy=initial_policy, MDP=training_MDP, max_iterations=np.inf)
        
        
        
        file_name = 'trained_array_wind_' + str(wind)
        file_name = file_name.replace('.', ',') # get rid of dots in file name
        np.save('results/4d/training_wind/' + file_name, trained_policy_array)


if __name__ == "__main__":
    evaluation_grid_size = 8
    evaluation_direction_prob = 0.7
    evaluation_obstacles = np.array([[3,2], [4,5], [6,3]], dtype='int32')
    evaluation_landing_zone = np.array([4,4], dtype='int32')
    evaluation_max_altitude = 10

    # direction_probability of evaluation_MDP doesn NOT matter in this process
    evaluation_MDP = dp4.MarkovGridWorld(grid_size=evaluation_grid_size, direction_probability=1, obstacles=evaluation_obstacles, landing_zone=evaluation_landing_zone, max_altitude=evaluation_max_altitude)
    train_policies(evaluation_MDP, 21)