import dynamic_programming_4d as dp4
import numpy as np

"""

DYNAMIC PROGRAMMING


"""

"""
BEGIN WITH PROBLEM USED IN TRAINING AND EVALUATING WITH DIFFERENT WIND PARAMETERS
"""
# keep in mind the direction_probability used here does NOT MATTER
wind_grid_size = 8
wind_obstacles = np.array([[3,2], [4,5], [6,3]], dtype='int32')
wind_landing_zone = np.array([4,4], dtype='int32')
wind_max_altitude = 10
wind_MDP = dp4.MarkovGridWorld(grid_size=wind_grid_size, direction_probability=1, obstacles=wind_obstacles, landing_zone=wind_landing_zone, max_altitude=wind_max_altitude)

# define parameters for training and evaluation
wind_eval_params = np.linspace(0,1,21)
wind_train_params = np.linspace(0,1,21)
wind_no_evaluations = 3000



"""

MONTE CARLO

"""

"""

PROBLEM USED IN CHOOSING EPSILON PARAMETER

"""
# must ensure numbers of episodes and policy improvement steps are NOT so big that most epsilons present indistinguishable performance!
epsilon_grid_size = 8
epsilon_obstacles = np.array([[3,2], [4,5], [6,3]], dtype='int32')
epsilon_landing_zone = np.array([4,4], dtype='int32')
epsilon_max_altitude = 10
epsilon_eval_wind = 0.90 # training and evaluating with equal wind levels.
epsilon_MDP = dp4.MarkovGridWorld(grid_size=epsilon_grid_size, direction_probability=epsilon_eval_wind, obstacles=epsilon_obstacles, landing_zone=epsilon_landing_zone, max_altitude=epsilon_max_altitude)
epsilon_no_episodes = 200
epsilon_no_steps = 5
epsilon_train_params = np.linspace(0, 1, 21)

epsilon_no_evaluations = 3000

"""

STUFF FOR CHOICE OF RATIO OF EPISODES TO SIZE OF STATE SPACE AND NUMBER OF POLICY IMPROVEMENT STEPS

"""

ratio_episodes_steps_epsilon = 0.2 # as per what was concluded in epsilon investigations

ratio_episodes_steps_ratio_params = np.linspace(0.1, 1, 10) # 0.1, 0.2, ..., 0.9, 1.0
ratio_episodes_steps_no_steps_params = np.linspace(1, 10, 10) # 1, 2, ..., 9, 10

ratio_episodes_steps_no_evaluations = 3000

"""

PROPER BENCHMARK PROBLEMS FOR OUR FINAL RESULTS

"""