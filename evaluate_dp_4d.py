import numpy as np
import dynamic_programming_4d as dp4
import monte_carlo_4d as mc4
import benchmark_problems_4d as bp4
import os
import matplotlib.pyplot as plt
import matplotlib

import fig_specs as fsp

# TRAINING WITH DIFFERENT WIND PARAMETERS AND SAVING POLICIES TO FILES
# NOTE, wind parameter of evaluation_MDP input does not matter. this will be set using values no_wind_parameters input argument values between 0 and 1.
def wind_train_policies(evaluation_MDP, wind_train_params):
    for wind in wind_train_params:
        wind = round(wind, 2) # round to 2 decimal places
        # training MDP same as evaluation MDP except for direction_probability
        training_MDP = dp4.MarkovGridWorld(grid_size=evaluation_MDP.grid_size, direction_probability=wind, obstacles=evaluation_MDP.obstacles, landing_zone=evaluation_MDP.landing_zone, max_altitude=evaluation_MDP.max_altitude)

        initial_policy = dp4.random_walk
        trained_policy, trained_policy_array = dp4.value_iteration(policy=initial_policy, MDP=training_MDP, max_iterations=np.inf)
        
        
        
        file_name = 'trained_array_wind_' + str(wind)
        file_name = file_name.replace('.', ',') # get rid of dots in file name
        np.save('results/4d/training_wind/' + file_name, trained_policy_array)



# EVALUATING POLICIES TRAINED WITH DIFFERENT WIND PARAMETERS SIMULATED IN MODELS WITH DIFFERENT MODELS
def wind_evaluate_policies(evaluation_MDP, no_evaluations, eval_wind_params, train_wind_params):
    # for each evaluation wind parameter, we have a column vector of the average scores of policies TRAINED with different wind parameters
    MDP = dp4.MarkovGridWorld(evaluation_MDP.grid_size, 1, evaluation_MDP.direction_probability, evaluation_MDP.obstacles, evaluation_MDP.landing_zone, evaluation_MDP.max_altitude)
    
    no_eval_wind_params = len(eval_wind_params)
    no_train_wind_params = len(train_wind_params)
    evaluations = np.zeros(shape=(no_train_wind_params, no_eval_wind_params))
    crashes = np.zeros(shape=(no_train_wind_params, no_eval_wind_params))
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
            crash_count = 0
            # run {no_evaluations} simulations using the fetched policy and record returns
            for _ in range(no_evaluations):

                history = mc4.generate_episode(MDP, policy)

                # BEWARE OF THE BELOW WHICH ONLY MAKES SENSE FOR RECIPROCAL OF L-1 NORM-TYPE REWARD
                if history[-1,-1] != 0:
                    cumulative_score += int(1 / history[-1,-1]) - 1
                else: # CRASHED
                    crash_count += 1
            average_score = cumulative_score / no_evaluations
            evaluations[i,j] = average_score
            crashes[i,j] = crash_count
            i += 1
            
            print(f'Another evaluation done. {to_go} more to go!')
            to_go -= 1
        j += 1
    return evaluations, crashes

# TO PLOT POLICY PERFORMANCE WITH VARYING WIND PARAMETERS
def wind_plot_evaluations(evaluations_array_txt_file_name, eval_wind_params, train_wind_params, surface=True, save=False):
    evaluations = np.loadtxt(evaluations_array_txt_file_name, ndmin=2)
    if not surface:
        plt.figure(figsize=(fsp.text_width, 2.5))

        for j in range(16, 20):
            plt.plot(train_wind_params, evaluations[:,j],  '-*', label='Evaluation wind parameter: ' + str(round(eval_wind_params[j],2)))

        plt.ylim(0, 3)
        plt.grid(True)
        plt.xlabel('Policy training wind parameter')
        plt.ylabel('Average landing error')
        plt.yticks(np.arange(0, 3.1, 1))
        plt.legend(ncol=2)
        plt.tight_layout()
        
    else:
        plot_x = train_wind_params
        plot_y = eval_wind_params
        X, Y = np.meshgrid(plot_x, plot_y)

        indices_x = np.arange(len(plot_x))
        indices_y = np.arange(len(plot_y))
        indices_x, indices_y = np.meshgrid(indices_x, indices_y)
        Z = evaluations [indices_x, indices_y]

        fig = plt.figure(figsize=(fsp.text_width * fsp.text_width_factor, fsp.fig_height))
        ax = fig.add_subplot(111, projection='3d')

        # plot evaluations
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)

        ax.set_xlabel('Training wind parameter')
        ax.set_ylabel('Evaluation wind parameter')
        ax.set_zlabel('Cost function performance')
        plt.grid(True)
        plt.title('Performance of policies trained and evaluated with differing wind parameters')
    if save:
        plt.savefig(f'{fsp.fig_path}/4d_dp_wind_evaluations.pdf')
    

def wind_plot_crash_rates(crashes_array_txt_file_name, no_evaluations, eval_wind_params, train_wind_params, surface = True, save=False):
    crashes = np.loadtxt(crashes_array_txt_file_name, ndmin=2)
    crash_rates = crashes / no_evaluations
    
    if not surface:
        no_eval_wind_params = len(eval_wind_params)
        no_mosaic_rows = 2

        plt.figure(figsize=(fsp.text_width, fsp.fig_height))

        #for j in range(no_eval_wind_params):
        for j in range(16, 20):
            j_sub = j - 16
            plt.subplot(no_mosaic_rows, int(np.ceil(4 / no_mosaic_rows)), j_sub+1)
            plt.plot(train_wind_params, crash_rates[:,j], 'b-*')
            
            plt.ylim(0, 1.1 * np.amax(crash_rates[:, 16:21]))
            plt.grid(True)
            plt.title('Evaluation wind parameter: ' + str(round(eval_wind_params[j],2)))
            plt.xlabel('Policy training wind parameter')
            plt.ylabel('Crash rate')
        plt.tight_layout()
    
    else:
        plot_x = train_wind_params
        plot_y = eval_wind_params
        X, Y = np.meshgrid(plot_x, plot_y)

        indices_x = np.arange(len(plot_x))
        indices_y = np.arange(len(plot_y))
        indices_x, indices_y = np.meshgrid(indices_x, indices_y)
        Z = crash_rates[indices_x, indices_y]

        fig = plt.figure(figsize=(fsp.text_width * fsp.text_width_factor, fsp.fig_height))
        ax = fig.add_subplot(111, projection='3d')

        # plot evaluations
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='cool', linewidth=0, antialiased=False)

        ax.set_xlabel('Training wind parameter')
        ax.set_ylabel('Evaluation wind parameter')
        ax.set_zlabel('Crash rate')
        plt.grid(True)
        plt.title('Crash rates of policies trained and evaluated with differing wind parameters')
    if save:
        plt.savefig(f'{fsp.fig_path}/4d_dp_wind_crash_rates.pdf')


# TO SAVE INFO ABOUT RESULTS FILE
def save_wind_results(evaluations_array, crashes_array, no_evaluations, eval_wind_params, train_wind_params, this_dir=True):
    if this_dir is False:
        evaluations_file_name = 'results/4d/training_wind/wind_evaluations_array.txt'
        crashes_file_name = 'results/4d/training_wind/wind_crashes_array.txt'
        info_file_name = 'results/4d/training_wind/wind_evaluations_info.txt'
    else:
        evaluations_file_name = 'wind_evaluations_array.txt'
        crashes_file_name = 'wind_crashes_array.txt'
        info_file_name = 'wind_evaluations_info.txt'
    confirmed = input(f'About to write results to {evaluations_file_name} and info to {info_file_name}. Proceed? (y/n)') == 'y'
    if confirmed:
        np.savetxt(evaluations_file_name, evaluations_array)
        np.savetxt(crashes_file_name, crashes_array)


        lines = [f'Information on the {evaluations_file_name} file.',
             '',
             'Number of simulations used per evaluation entry: ' + str(no_evaluations),
             'Evaluated policies were trained with the following wind parameters: ' + str(train_wind_params),
             'Policies were evaluated in MDPs with the following wind parameters: ' + str(eval_wind_params)]
    
        with open(info_file_name, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        print(f'Results and info written to {evaluations_file_name} and {info_file_name}.')
    else:
        print('Did not confirm. Files NOT written.')

# FOR EVALUATING RANDOM WALK ONLY 
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

# TRAINING MULTIPLE MONTE CARLO POLICIES WITH VARYING EPSILON
def epsilon_train_policies(MDP, epsilon_params, evaluation_no_episodes, no_improvement_steps):
    for epsilon in epsilon_params:
        epsilon = round(epsilon, 2) # round to 2 decimal places

        initial_policy = dp4.random_walk
        trained_policy, trained_policy_array = mc4.monte_carlo_policy_iteration(initial_policy, MDP, epsilon, evaluation_no_episodes, no_improvement_steps)
        
        file_name = 'trained_array_epsilon_' + str(epsilon)
        file_name = file_name.replace('.', ',') # get rid of dots in file name
        np.save('results/4d/training_epsilon/' + file_name, trained_policy_array)

def epsilon_evaluate_policies(MDP, no_evaluations, epsilon_train_params):
    no_train_epsilon_params = len(epsilon_train_params)
    evaluations = np.zeros(shape=no_train_epsilon_params)
    crashes = np.zeros(shape=no_train_epsilon_params)
    to_go = no_train_epsilon_params
    
    i = 0
    for train_epsilon in epsilon_train_params:
        train_epsilon = round(train_epsilon, 2)

        # fetch the corresponding pre-computed policy
        file_name = 'results/4d/training_epsilon/trained_array_epsilon_' + str(train_epsilon)
        file_name = file_name.replace('.', ',')
        file_name = file_name + '.npy'
        policy_array = np.load(file_name)
        policy = dp4.array_to_policy(policy_array, MDP)

        cumulative_score = 0
        crash_count = 0
        # run {no_evaluations} simulations using the fetched policy and record returns
        for _ in range(no_evaluations):

            history = mc4.generate_episode(MDP, policy)

            # BEWARE OF THE BELOW WHICH ONLY MAKES SENSE FOR RECIPROCAL OF L-1 NORM-TYPE REWARD
            if history[-1,-1] != 0:
                cumulative_score += int(1 / history[-1,-1]) - 1
            else: # CRASHED
                crash_count += 1
        average_score = cumulative_score / no_evaluations
        evaluations[i] = average_score
        crashes[i] = crash_count
        i += 1
        
        print(f'Another evaluation done. {to_go} more to go!')
        to_go -= 1
    return evaluations, crashes

def save_epsilon_results(evaluations_array, crashes_array, no_evaluations, train_epsilon_params, epsilon_wind_param, this_dir=True):
    if this_dir is False:
        evaluations_file_name = 'results/4d/training_epsilon/epsilon_evaluations_array.txt'
        crashes_file_name = 'results/4d/training_epsilon/epsilon_crashes_array.txt'
        info_file_name = 'results/4d/training_epsilon/epsilon_evaluations_info.txt'
    else:
        evaluations_file_name = 'epsilon_evaluations_array.txt'
        crashes_file_name = 'epsilon_crashes_array.txt'
        info_file_name = 'epsilon_evaluations_info.txt'
    confirmed = input(f'About to write results to {evaluations_file_name} and info to {info_file_name}. Proceed? (y/n)') == 'y'
    if confirmed:
        np.savetxt(evaluations_file_name, evaluations_array)
        np.savetxt(crashes_file_name, crashes_array)


        lines = [f'Information on the {evaluations_file_name} file.',
             '',
             'Number of simulations used per evaluation entry: ' + str(no_evaluations),
             'Evaluated policies were trained with the following epsilon parameters: ' + str(train_epsilon_params),
             'Policies were evaluated and trained in MDPs with the following wind parameter: ' + str(epsilon_wind_param)]
    
        with open(info_file_name, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        print(f'Results and info written to {evaluations_file_name} and {info_file_name}.')
    else:
        print('Did not confirm. Files NOT written.')

def epsilon_plot_evaluations(evaluations_array_txt_file_name, train_epsilon_params, save=False):
    evaluations = np.loadtxt(evaluations_array_txt_file_name, ndmin=1)

    plt.figure(figsize=(fsp.text_width * fsp.text_width_factor, fsp.fig_height))
    plt.plot(train_epsilon_params, evaluations[:], 'r-*')
    
    #plt.ylim(np.amin(evaluations), 0)
    plt.ylim(0, np.amax(evaluations))
    plt.grid(True)
    plt.title('Performance of MC policies trained with different epsilon parameters')
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{fsp.fig_path}/4d_mc_epsilon_evaluations.pdf')
    
    #plt.show()


def epsilon_plot_crash_rates(crashes_array_txt_file_name, no_evaluations, train_epsilon_params, save=False):
    crashes = np.loadtxt(crashes_array_txt_file_name, ndmin=1)
    crash_rates = crashes / no_evaluations

    plt.figure(figsize=(fsp.text_width * fsp.text_width_factor, fsp.fig_height))
    plt.plot(train_epsilon_params, crash_rates[:], 'b-*')
    
    plt.grid(True)
    plt.title('Crash rates of MC policies trained with different epsilon parameters')
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{fsp.fig_path}/4d_mc_epsilon_crash_rates.pdf')
    
    #plt.show()



if __name__ == "__main__":
    os.system('clear')
    
    """
    
    WIND PARAMETER CHOICE
    
    """
    # TRAIN POLICIES AT DIFFERENT WIND PARAMATERS AND STORE IN RESULTS DIRECTORY.
    wind_train = False
    if wind_train:
        wind_train_policies(bp4.wind_MDP, bp4.wind_train_params) # 21 wind training parameters

    # EVALUATE POLICIES TRAINED WITH DIFFERENT WIND PARAMETERS SIMULATED IN PROBLEMS WITH DIFFERENT WIND PARAMETERS,
    # STORE 2D ARRAY RESULTS IN RESULTS DIRECTORY.
    wind_evaluate = False
    if wind_evaluate:
        evaluations, crashes = wind_evaluate_policies(bp4.wind_MDP, bp4.wind_no_evaluations, bp4.wind_eval_params, bp4.wind_train_params)
        #evaluations, crashes = wind_evaluate_policies(bp4.wind_MDP, 10, bp4.wind_eval_params, bp4.wind_train_params)
        print(f'Evaluated policies using {bp4.wind_no_evaluations} simulations each.')
        print(f'Evaluated policies trained with following wind parameters (each row corresponds to a policy): {bp4.wind_train_params}')
        print(f'Evaluated policies using MDPs with following wind parameters (each column corresponds to an evaluation MDP): {bp4.wind_eval_params}')
        print('Evaluations array:')
        print(evaluations)
        print('Crashes array:')
        print(crashes)
        save_wind_results(evaluations, crashes, bp4.wind_no_evaluations, bp4.wind_eval_params, bp4.wind_train_params, this_dir=True)
    
    # READ 2D ARRAY RESULTS FROM PREVIOUS STEP (ABOUT CHOOSING WIND PARAMETER) AND PLOT THEM.
    wind_plot = True
    if wind_plot:
        evaluations_file = 'results/4d/training_wind/wind_evaluations_array.txt'
        crashes_file = 'results/4d/training_wind/wind_crashes_array.txt'
        wind_plot_evaluations(evaluations_file, bp4.wind_eval_params, bp4.wind_train_params, surface=False, save = False)
        wind_plot_crash_rates(crashes_file, bp4.wind_no_evaluations, bp4.wind_eval_params, bp4.wind_train_params, surface = False, save = False)
        plt.show()







    """
    epsilon_train = False
    if epsilon_train:
        epsilon_train_policies(bp4.epsilon_MDP, bp4.epsilon_train_params, bp4.epsilon_no_episodes, bp4.epsilon_no_steps)

    epsilon_evaluate = False
    if epsilon_evaluate:
        evaluations, crashes = epsilon_evaluate_policies(bp4.epsilon_MDP, bp4.epsilon_no_evaluations, bp4.epsilon_train_params)
        print(f'Evaluated policies using {bp4.epsilon_no_evaluations} simulations each.')
        print(f'Evaluated policies trained with following epsilon parameters: {bp4.epsilon_train_params}')
        print('Evaluations array:')
        print(evaluations)
        print('Crashes array:')
        print(crashes)
        save_epsilon_results(evaluations, crashes, bp4.wind_no_evaluations, bp4.wind_eval_params, bp4.wind_train_params, this_dir=False)

    epsilon_plot = False
    if epsilon_plot:
        evaluations_file = 'results/4d/training_epsilon/epsilon_evaluations_array.txt'
        crashes_file = 'results/4d/training_epsilon/epsilon_crashes_array.txt'
        epsilon_plot_evaluations(evaluations_file, bp4.epsilon_train_params, save = False)
        epsilon_plot_crash_rates(crashes_file, bp4.epsilon_no_evaluations, bp4.epsilon_train_params, save = False)
    """