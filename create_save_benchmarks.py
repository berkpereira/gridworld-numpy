import numpy as np
import os
import dynamic_programming_4d as dp4
import dynamic_programming_3d as dp3
import pickle

# define parameters
grid_sizes = range(4, 14)
IDs = range(1, 5)
wind_params = np.arange(0.70, 1.02, 0.05)


# IN ORDER TO CHANGE BETWEEN 4D AND 3D, change the below variable
dimension = 3

for grid_size in grid_sizes:
    for ID in IDs:
        # load obstacles array
        obstacles_file = f"benchmarks-aux/obstacles/obstacles_array_{grid_size}{ID}.npy"
        obstacles = np.load(obstacles_file)

        # load landing zone array
        landing_zone_file = f"benchmarks-aux/landing-zones/landing_zone_array_{grid_size}{ID}.npy"
        landing_zone = np.load(landing_zone_file)

        for wind_param in wind_params:
            # create class instance
            if dimension == 4:
                MDP = dp4.MarkovGridWorld(grid_size=grid_size, discount_factor=1, direction_probability=wind_param,
                                        obstacles=obstacles, landing_zone=landing_zone, max_altitude=grid_size * 2)
            elif dimension == 3:
                MDP = dp3.MarkovGridWorld(grid_size=grid_size, discount_factor=1, direction_probability=wind_param,
                                        obstacles=obstacles, landing_zone=landing_zone, max_altitude=grid_size * 2)
            else:
                Exception("Invalid dimension! Must be 3 or 4.")

            # save class instance to file
            if dimension == 4:
                filename = f"benchmark-problems/4d/{grid_size}{ID}_wind_{str(round(wind_param,2)).replace('.', ',')}.p"
            elif dimension == 3:
                filename = f"benchmark-problems/3d/{grid_size}{ID}_wind_{str(round(wind_param,2)).replace('.', ',')}.p"
            else:
                Exception("Invalid dimension! Must be 3 or 4.")

            with open(filename, "wb") as f:
                pickle.dump(MDP, f)
