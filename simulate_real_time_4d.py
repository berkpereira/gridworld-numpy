import numpy as np

import dynamic_programming_4d as dp4
import solve_mip4d as mip4
import monte_carlo_4d as mc4

# this will take a sequence of actions and expected outcomes, simulate time steps by sampling MDP dynamics, and do *something* if the real outcomes
# deviate from the expected ones at any point (this is necessary in the MIP case, since a new solution would need to be computed)
def simulate_closed_loop():
    pass


if __name__ == "__main__":
    MDP_list = [dp4.MarkovGridWorld(grid_size=5, direction_probability=1, obstacles=np.array([[]]), landing_zone=np.array([0,0]), max_altitude=10),
                dp4.MarkovGridWorld(grid_size=4, direction_probability=1, obstacles=np.array([[]]), landing_zone=np.array([0,0]), max_altitude=5)]
    
    
    