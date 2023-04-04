# this file serves as a "playground" for miscellaneous animating of agent policies and MIP solutions

import monte_carlo_4d as mc4
import dynamic_programming_4d as dp4
import solve_mip4d as mip4
import numpy as np
import copy
import os
import pickle


if __name__ == "__main__":
    MDP = dp4.MarkovGridWorld(grid_size = 5, obstacles=np.array([[2,3], [1,1]]), landing_zone=np.array([3,2]), max_altitude=8)
    history = mc4.generate_episode(MDP, dp4.random_walk)
    mc4.play_episode(MDP, dp4.random_walk, history, save = True)
