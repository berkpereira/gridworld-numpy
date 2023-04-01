# SIMPLE AUXILIARY SCRIPT TO CREATE 2D ARRAYS OF OBSTACLES FOR MDP BENCHMARK PROBLEMS.
# RUN IT AND IT WILL SAVE A 2D ARRAY OF OBSTACLES TO A FILE.
import numpy as np
import dynamic_programming_4d as dp4
import monte_carlo_4d as mc4
import dynamic_programming_3d as dp3
import monte_carlo_3d as mc3

def generate_obstacles(grid_size, landing_zone, no_obstacles):
    # Generate possible obstacle positions
    possible_obstacles = np.array(np.meshgrid(np.arange(grid_size), np.arange(grid_size))).T.reshape(-1, 2)

    # Remove landing zone position from possible obstacles
    possible_obstacles = possible_obstacles[~np.all(possible_obstacles == landing_zone, axis=1)]

    # Randomly select unique obstacles
    indices = np.random.choice(len(possible_obstacles), size=no_obstacles, replace=False)
    obstacles = possible_obstacles[indices]

    return obstacles



import numpy as np

IDs = np.array([41, 42, 43, 44, 51, 52, 53, 54, 61, 62, 63, 64, 71, 72, 73, 74, 81, 82, 83, 84, 91, 92, 93, 94, 101, 102, 103, 104, 111, 112, 113, 114, 121, 122, 123, 124, 131, 132, 133, 134], dtype='int32')
grid_sizes = np.array([4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13], dtype='int32')
max_alts = np.array([8, 8, 8, 8, 10, 10, 10, 10, 12, 12, 12, 12, 14, 14, 14, 14, 16, 16, 16, 16, 18, 18, 18, 18, 20, 20, 20, 20, 22, 22, 22, 22, 24, 24, 24, 24, 26, 26, 26, 26], dtype='int32')
landing_zones = np.array([[2,2], [2,2], [1,1], [2,2], # 4
                          [2,2], [2,2], [2,2], [2,2], # 5
                          [4,2], [2,3], [3,2], [4,2], # 6
                          [5,3], [3,4], [3,3], [5,3], # 7
                          [2,3], [3,4], [3,5], [2,3], # 8
                          [4,3], [3,4], [4,6], [4,3], # 9
                          [4,4], [5,3], [7,3], [4,4], # 10
                          [6,5], [4,6], [5,4], [6,5], # 11
                          [4,6], [5,4], [7,6], [4,6], # 12
                          [6,6], [7,7], [4,5], [6,6]], dtype='int32') # 13


set_buildings = [[], [], [[2,2]], [[1,1], [2,1]], # 4, edge case here with 2 empty-obstacle problems to begin with. let it be
                 [], [[1,1]], [[1,1], [3,2]], [[1,1], [3,2], [3,3]], # 5
                 [], [[2,1]], [[1,3], [2,4], [3,4]], [[2,1], [3,1], [2,4], [1,4], [5,1]], # 6
                 [], [[5,2], [1,5]], [[1,1], [1,5], [5,2], [5,4]], [[0,0], [1,0], [1,3], [2,5], [3,1], [5,5], [6,1]], # 7
                 [], [[1,6], [5,2], [6,5]], [[1,3], [1,6], [3,1], [5,1], [6,3], [6,6]], [[0,7], [1,1], [1,7], [2,5], [4,0], [5,6], [6,1], [6,2], [6,4]], # 8
                 [], [[1,7], [2,1], [5,6], [6,2]], [[1,1], [1,5], [1,7], [3,2], [5,0], [6,6], [7,1], [7,3]], [[1,1], [1,3], [1,7], [2,5], [4,0], [4,8], [5,6], [7,0], [7,2], [7,8], [8,0], [8,5]], # 9
                 [], [[1,3], [2,7], [6,5], [8,1], [8,8]], [[0,3], [1,5], [1,8], [2,1], [4,4], [4,7], [5,1], [7,6], [8,1], [8,8]], [[0,2], [0,7], [1,0], [1,9], [2,3], [2,5], [3,7], [5,1], [5,9], [7,3], [7,6], [8,0], [8,9], [9,0], [9,9]], # 10
                 [], [[1,3], [1,7], [5,4], [5,9], [7,1], [9,5]], [[1,1], [1,6], [2,3], [2,8], [4,0], [4,9], [5,6], [6,1], [7,8], [8,4], [9,2], [9,6]], [[0,5], [1,2], [1,9], [2,0], [2,5], [3,3], [3,7], [4,9], [5,0], [6,3], [7,0], [7,7], [7,10], [9,2], [9,5], [9,8], [10,0], [10,2]], # 11
                 [], [[2,2], [2,6], [3,9], [6,1], [6,7], [9,2], [9,6]], [[1,7], [1,10], [2,3], [2,5], [4,1], [4,8], [5,4], [7,10], [8,1], [8,2], [10,1], [10,4], [10,6], [10,9]], [[1,3], [1,6], [1,10], [2,1], [2,8], [3,4], [4,10], [5,0], [5,3], [5,8], [7,2], [7,7], [8,5], [8,10], [8,11], [9,0], [9,1], [10,4], [10,6], [11,8], [11,9]], # 12
                 [], [[1,9], [3,2], [3,6], [6,4], [6,11], [8,2], [10,4], [11,9]], [[1,1], [1,3], [1,6], [1,11], [3,10], [4,1], [5,8], [6,1], [6,6], [7,11], [8,4], [9,1], [9,7], [10,9], [11,4], [11,11]], [[0,11], [0,12], [1,3], [1,5], [2,1], [2,8], [2,9], [3,6], [3,11], [4,0], [5,0], [5,3], [6,9], [6,12], [8,0], [8,2], [8,4], [9,6], [9,9], [10,11], [11,1], [11,4], [11,5], [11,7], [11,11]]] # 13


no_buildings = np.zeros(len(IDs), dtype='int32')

# Set numbers of building ratios
buildings_ratios = [0, 0.05, 0.1, 0.15]
i = 0
for _ in no_buildings:
    no_buildings[i] = int(np.floor(buildings_ratios[i % 4] * (grid_sizes[i] ** 2)))
    i += 1

print(no_buildings)

input('Press enter...')
for i in range(len(IDs)):
    # Define inputs
    MDP_ID = IDs[i]
    grid_size = grid_sizes[i]
    landing_zone = landing_zones[i]
    no_obstacles = no_buildings[i]

    # Generate obstacles
    #obstacles = generate_obstacles(grid_size, landing_zone, no_obstacles)
    obstacles = np.array(set_buildings[i], ndmin=2)

    # Print obstacles
    print(obstacles)
    print()
    print(landing_zone)
    print()
    print()

    #potential_MDP = dp4.MarkovGridWorld(grid_size=grid_size, obstacles=obstacles, landing_zone=landing_zone, max_altitude=grid_size*2)
    #history = mc4.generate_episode(potential_MDP, dp4.random_walk)
    #mc4.play_episode(potential_MDP, dp4.random_walk, history)

    # Save obstacles
    save_obstacles = input('Save obstacles? (y/n)') == 'y'
    #save_obstacles = True
    if save_obstacles:
        obstacles_file_name = "benchmarks-aux/obstacles/obstacles_array_" + str(MDP_ID)
        np.save(obstacles_file_name, obstacles)

    # Save landing zones
    save_landing_zone = input('Save landing zone? (y/n)') == 'y'
    #save_landing_zone = True
    if save_landing_zone:
        landing_zone_file_name = "benchmarks-aux/landing-zones/landing_zone_array_" + str(MDP_ID)
        np.save(landing_zone_file_name, landing_zone)