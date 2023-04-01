# this file serves as a "playground" for miscellaneous animating of agent policies and MIP solutions

import monte_carlo_4d as mc4
import dynamic_programming_4d as dp4
import solve_mip4d as mip4
import numpy as np
import copy
import os
import pickle


if __name__ == "__main__":
    import os

    dir_path = 'benchmark-problems/4d/'
    files = sorted(os.listdir(dir_path))

    # Create an empty list to store the objects
    objects = []

    # Iterate over the files in alphabetical order
    for filename in files:
        # Open the file in binary mode
        with open(os.path.join(dir_path, filename), "rb") as file:
            # Deserialize the object from the file and append it to the list
            obj = pickle.load(file)
            objects.append(obj)

    for MDP in objects:
        history = mc4.generate_episode(MDP, dp4.random_walk)
        mc4.play_episode(MDP, dp4.random_walk, history)
