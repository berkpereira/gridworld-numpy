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
MDP_ID = 133
grid_size = 4
landing_zone = np.array([3, 3])
no_obstacles = 6

# Generate obstacles
obstacles = generate_obstacles(grid_size, landing_zone, no_obstacles)

# Print obstacles
print(obstacles)

# Save obstacles, subject to approval
#save = input('Save obstacles array to file? (y/n)') == 'y'
save_obstacles = False
if save_obstacles:
    obstacles_file_name = "benchmarks-aux/obstacles/obstacles_array_" + str(MDP_ID)
    np.save(obstacles_file_name, obstacles)

save_landing_zone = True
if save_landing_zone:
    landing_zone_file_name = "benchmarks-aux/landing-zones/landing_zone_array_" + str(MDP_ID)
    np.save(landing_zone_file_name, landing_zone)