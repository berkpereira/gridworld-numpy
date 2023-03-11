import monte_carlo_4d as mc4
import dynamic_programming_4d as dp4
import numpy as np



def evaluate_policy(policy, evaluation_MDP, no_evaluations):
    cumulative_score = 0
    for _ in range(no_evaluations):
        history = mc4.generate_episode(evaluation_MDP, policy)
        cumulative_score += history[-1,-1]
    average_score = cumulative_score / no_evaluations
    return average_score





if __name__ == "__main__":
    evaluation_grid_size = 8
    evaluation_direction_prob = 1
    evaluation_obstacles = np.array([[3,2], [4,5], [6,3]], dtype='int32')
    evaluation_landing_zone = np.array([4,4], dtype='int32')
    evaluation_max_altitude = 10
    evaluation_MDP = dp4.MarkovGridWorld(grid_size=evaluation_grid_size, direction_probability=evaluation_direction_prob, obstacles=evaluation_obstacles, landing_zone=evaluation_landing_zone, max_altitude=evaluation_max_altitude)

    policy_scores = np.zeros(shape=21)
    i = 0
    for wind in np.linspace(0, 1, 21):
        wind = round(wind, 2)
        file_name = 'results/4d/training_wind/trained_array_wind_' + str(wind)
        file_name = file_name.replace('.', ',')
        file_name = file_name + '.npy'
        policy_array = np.load(file_name)
        policy = dp4.array_to_policy(policy_array, evaluation_MDP)

        no_evaluations = 5000
        average_policy_score = -1 * evaluate_policy(policy, evaluation_MDP, no_evaluations)
        policy_scores[i] = average_policy_score
        print(f'Policy training "wind": {wind}')
        print(f'Average score obtained over {no_evaluations} evaluations: {average_policy_score}')
        #input('Press enter to continue...')
        i += 1
    np.savetxt('wind_evaluation_results.txt', policy_scores)




