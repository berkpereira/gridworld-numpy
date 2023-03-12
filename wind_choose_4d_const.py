import dynamic_programming_4d as dp4
import numpy as np

# define the evaluation MDP.
# keep in mind the direction_probability used here does NOT MATTER


evaluation_grid_size = 8
evaluation_obstacles = np.array([[3,2], [4,5], [6,3]], dtype='int32')
evaluation_landing_zone = np.array([4,4], dtype='int32')
evaluation_max_altitude = 10
evaluation_MDP = dp4.MarkovGridWorld(grid_size=evaluation_grid_size, direction_probability=1, obstacles=evaluation_obstacles, landing_zone=evaluation_landing_zone, max_altitude=evaluation_max_altitude)

# define parameters for training and evaluation
eval_wind_params = np.linspace(0,1,21)
train_wind_params = np.linspace(0,1,21)
no_evaluations = 3000