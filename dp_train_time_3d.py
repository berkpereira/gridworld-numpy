import dynamic_programming_3d as dp3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_training_time(train_wind_param, env_aspect_ratio, grid_sizes):
    training_times = np.zeros(shape=len(grid_sizes))
    for i in range(len(grid_sizes)):
        grid_size = grid_sizes[i]
        MDP = dp3.MarkovGridWorld(grid_size=grid_size, direction_probability=train_wind_param,
                                  obstacles=np.array([], ndmin=2), landing_zone=np.array([0,0], dtype='int32'),
                                  max_altitude=int(np.ceil(env_aspect_ratio * grid_size)))
        
        trained_policy, trained_policy_array, train_time = dp3.value_iteration(policy=dp3.random_walk,
                                                                                  MDP=MDP, max_iterations=np.inf, train_time=True)
        training_times[i] = train_time
    
    training_df = pd.DataFrame({"grid_size":grid_sizes, "training_time":training_times})
    training_df.set_index("grid_size", drop=True, inplace=True)
    return training_df
    
        
def plot_train_times(train_times_df, save=False):
    fig, ax = plt.subplots()
    ax.set_xlabel("grid size")
    ax.set_ylabel("training time (seconds)")
    fig.set_size_inches(16,9.5)
    plt.plot(train_times_df["training_time"], 'r--*')
    plt.grid(True)
    plt.title('Training time vs grid size (3D state space)')
    plt.tight_layout()

    if save:
        plt.savefig('out_plot.pdf')
    
    plt.show()



if __name__ == "__main__":
    grid_sizes = list(range(1,3))
    aspect_ratio = 2
    wind_param = 0.90
    train_times = get_training_time(train_wind_param=wind_param, env_aspect_ratio=aspect_ratio, grid_sizes=grid_sizes)
    print(train_times)
    print()
    print(train_times["training_time"])
    
    plot_train_times(train_times, save=True)

    #train_times.to_csv('train_times.csv', index=True)
    #pd.read_csv('train_times.csv', index_col="grid_size") # including correct indices for easy plotting 