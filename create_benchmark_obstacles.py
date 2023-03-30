# SIMPLE AUXILIARY SCRIPT TO CREATE 2D ARRAYS OF OBSTACLES FOR MDP BENCHMARK PROBLEMS.
# RUN IT AND IT WILL SAVE A 2D ARRAY OF OBSTACLES TO A FILE.
import numpy as np

def generate_obstacles(grid_size, landing_zone, no_obstacles):
    # Generate possible obstacle positions
    possible_obstacles = np.array(np.meshgrid(np.arange(grid_size), np.arange(grid_size))).T.reshape(-1, 2)

    # Remove landing zone position from possible obstacles
    possible_obstacles = possible_obstacles[~np.all(possible_obstacles == landing_zone, axis=1)]

    # Randomly select unique obstacles
    indices = np.random.choice(len(possible_obstacles), size=no_obstacles, replace=False)
    obstacles = possible_obstacles[indices]

    return obstacles


# Define inputs
MDP_ID = 
grid_size = 
landing_zone = np.array([4, 4])
no_obstacles = 

# Generate obstacles
obstacles = generate_obstacles(grid_size, landing_zone, no_obstacles)

# Print obstacles
print(obstacles)

# Save obstacles, subject to approval
#save = input('Save obstacles array to file? (y/n)') == 'y'
save = True
if save:
    obstacles_file_name = "benchmarks-aux/obstacles/obstacles_array_" + str(MDP_ID)
    np.save(obstacles_file_name, obstacles)
