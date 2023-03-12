import monte_carlo_4d as mc4
import dynamic_programming_4d as dp4
import numpy as np
import os
import matplotlib.pyplot as plt



def evaluate_policy_winds(evaluation_MDP, no_evaluations, eval_wind_params, train_wind_params):
    # for each evaluation wind parameter, we have a column vector of the average scores of policies TRAINED with different wind parameters
    MDP = dp4.MarkovGridWorld(evaluation_MDP.grid_size, 1, evaluation_MDP.direction_probability, evaluation_MDP.obstacles, evaluation_MDP.landing_zone, evaluation_MDP.max_altitude)
    
    no_eval_wind_params = len(eval_wind_params)
    no_train_wind_params = len(train_wind_params)
    evaluations = np.zeros(shape=(no_train_wind_params, no_eval_wind_params))
    to_go = no_train_wind_params * no_eval_wind_params
    j = 0
    for eval_wind in eval_wind_params:
        eval_wind = round(eval_wind, 2)

        # set up MDP used to run simulations
        MDP = dp4.MarkovGridWorld(evaluation_MDP.grid_size, 1, eval_wind, evaluation_MDP.obstacles, evaluation_MDP.landing_zone, evaluation_MDP.max_altitude)
        i = 0
        for train_wind in train_wind_params: # saying cases with direction_probability < 0.50 are unreasonable!
            train_wind = round(train_wind, 2)

            # fetch the corresponding pre-computed policy
            file_name = 'results/4d/training_wind/trained_array_wind_' + str(train_wind)
            file_name = file_name.replace('.', ',')
            file_name = file_name + '.npy'
            policy_array = np.load(file_name)
            policy = dp4.array_to_policy(policy_array, MDP)

            cumulative_score = 0
            # run {no_evaluations} simulations using the fetched policy and record returns
            for _ in range(no_evaluations):

                history = mc4.generate_episode(MDP, policy)
                cumulative_score += history[-1,-1]
            average_score = cumulative_score / no_evaluations
            evaluations[i,j] = average_score    
            i += 1
            
            print(f'Another evaluation done. {to_go} more to go!')
            to_go -= 1
        j += 1
    return evaluations

def plot_wind_evaluations(evaluations_array_txt_file_name, eval_wind_params, train_wind_params, save=False):
    evaluations = np.loadtxt(evaluations_array_txt_file_name, ndmin=2)
    no_eval_wind_params = len(eval_wind_params)
    no_mosaic_rows = 3

    plt.figure(figsize=(12,9))

    for j in range(no_eval_wind_params):
        plt.subplot(no_mosaic_rows, int(np.ceil(no_eval_wind_params / no_mosaic_rows)), j+1)
        plt.plot(train_wind_params, evaluations[:,j], 'r-*')
        #plt.ylim(np.amin(evaluations), 0)
        plt.ylim(0, 0.8)
        plt.grid(True)
        plt.title('Evaluation wind: ' + str(round(eval_wind_params[j],2)))
    plt.tight_layout()
    
    if save:
        plt.savefig('out_plot.pdf')
    
    plt.show()

    
    
def save_results_info(evaluations_array, no_evaluations, eval_wind_params, train_wind_params, this_dir=True):
    if this_dir is False:
        results_file_name = 'results/4d/training_wind/wind_evaluations_array.txt'
        info_file_name = 'results/4d/training_wind/wind_evaluations_info.txt'
    else:
        results_file_name = 'wind_evaluations_array.txt'
        info_file_name = 'wind_evaluations_info.txt'
    confirmed = input(f'About to write results to {results_file_name} and info to {info_file_name}. Proceed? (y/n)') == 'y'
    if confirmed:
        np.savetxt(results_file_name, evaluations_array)

        lines = [f'Information on the {results_file_name} file.',
             '',
             'Number of simulations used per evaluation entry: ' + str(no_evaluations),
             'Evaluated policies were trained with the following wind parameters: ' + str(train_wind_params),
             'Policies were evaluated in MDPs with the following wind parameters: ' + str(eval_wind_params)]
    
        with open(info_file_name, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        print(f'Results and info written to {results_file_name} and {info_file_name}.')
    else:
        print('Did not confirm. Files NOT written.')

def evaluate_random_walk(evaluation_MDP, no_evaluations, eval_wind_params):
    # for each evaluation wind parameter, we have a column vector of the average scores of policies TRAINED with different wind parameters
    MDP = dp4.MarkovGridWorld(evaluation_MDP.grid_size, 1, evaluation_MDP.direction_probability, evaluation_MDP.obstacles, evaluation_MDP.landing_zone, evaluation_MDP.max_altitude)
    
    no_eval_wind_params = len(eval_wind_params)
    evaluations = np.zeros(shape=(1, no_eval_wind_params))
    j = 0
    for eval_wind in eval_wind_params:
        eval_wind = round(eval_wind, 2)

        # set up MDP used to run simulations
        MDP = dp4.MarkovGridWorld(evaluation_MDP.grid_size, 1, eval_wind, evaluation_MDP.obstacles, evaluation_MDP.landing_zone, evaluation_MDP.max_altitude)
        
        policy = dp4.random_walk


        cumulative_score = 0
        # run {no_evaluations} simulations using the fetched policy and record returns
        for _ in range(no_evaluations):

            history = mc4.generate_episode(MDP, policy)
            cumulative_score += history[-1,-1]
        average_score = cumulative_score / no_evaluations
        evaluations[0,j] = average_score    
        
        j += 1
    return evaluations



if __name__ == "__main__":
    os.system('clear')

    import wind_choose_4d_const as wconst4
    want_evaluate = True


    if want_evaluate:
        evaluations = evaluate_policy_winds(wconst4.evaluation_MDP, wconst4.no_evaluations, wconst4.eval_wind_params, wconst4.train_wind_params)
        print(f'Evaluated policies using {wconst4.no_evaluations} simulations each.')
        print(f'Evaluated policies trained with following wind parameters (each row corresponds to a policy): {wconst4.train_wind_params}')
        print(f'Evaluated policies using MDPs with following wind parameters (each column corresponds to an evaluation MDP): {wconst4.eval_wind_params}')
        print(evaluations)
        save_results_info(evaluations, wconst4.no_evaluations, wconst4.eval_wind_params, wconst4.train_wind_params, this_dir=True)


    evaluations_file = 'results/4d/training_wind/random_walk_evaluation_array.txt'
    evaluations_file = 'wind_evaluations_array.txt'
    plot_wind_evaluations(evaluations_file, wconst4.eval_wind_params, wconst4.train_wind_params, True)



