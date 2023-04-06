import numpy as np
import dynamic_programming_4d as dp4
import monte_carlo_4d as mc4
import benchmark_problems_4d as bp4
import os
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import fig_specs as fsp

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

    plt.figure(figsize=(fsp.text_width * fsp.text_width_factor, 3.1))
    plt.plot(train_epsilon_params, evaluations[:], '-*')
    
    plt.ylim(1.5, 1.05 * np.amax(evaluations))
    plt.grid(True)
    #plt.title('Performance of MC policies trained with different epsilon parameters')
    plt.yticks([1.5, 2, 2.5, 3])
    plt.xlabel('Policy training ε parameter')
    plt.ylabel('Average landing error')

    plt.tight_layout()
    
    if save:
        plt.savefig(f'{fsp.fig_path}/4d-mc-epsilon-evaluations.pdf')
    
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

# TRAIN POLICIES USING DIFFERENT RATIO OF EPISODES TO STATE SPACE SIZE AND DIFFERENT NUMBER OF IMPROVEMENT STEPS
def ratio_steps_train_policies(MDP, no_episodes_ratio_params, no_steps_params):
    for no_episodes_ratio in no_episodes_ratio_params:
        no_episodes_ratio = round(no_episodes_ratio, 2) # round to 2 decimal places
        no_episodes = int(np.ceil(no_episodes_ratio * MDP.state_space.shape[0])) # actual number of episodes to use

        for no_steps in no_steps_params:
            no_steps = int(no_steps)

            initial_policy = dp4.random_walk
            trained_policy, trained_policy_array = mc4.monte_carlo_policy_iteration(initial_policy, MDP, bp4.ratio_episodes_steps_epsilon, no_episodes, no_steps)
            
            file_name = 'trained_array_ratio_' + str(no_episodes_ratio) + '_steps_' + str(no_steps)
            file_name = file_name.replace('.', ',') # get rid of decimal dots in file name
            np.save('results/4d/training_ratio_steps/' + file_name, trained_policy_array)

def ratio_steps_evaluate_policies(MDP, no_evaluations, no_episodes_ratio_params, no_steps_params):
    no_no_episodes_ratio_params = len(no_episodes_ratio_params)
    no_no_steps_params = len(no_steps_params)
    evaluations = np.zeros(shape=(no_no_episodes_ratio_params, no_no_steps_params))
    crashes = np.zeros(shape=(no_no_episodes_ratio_params, no_no_steps_params))
    
    to_go = no_no_episodes_ratio_params * no_no_steps_params
    
    i = 0
    for no_episodes_ratio in no_episodes_ratio_params:
        no_episodes_ratio = round(no_episodes_ratio, 2)

        j = 0
        for no_steps in no_steps_params:
            no_steps = int(no_steps)

            # fetch the corresponding pre-computed policy
            file_name = 'results/4d/training_ratio_steps/trained_array_ratio_' + str(no_episodes_ratio) + '_steps_' + str(no_steps)
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
            evaluations[i, j] = average_score
            crashes[i, j] = crash_count
            
            j += 1
            print(f'Another evaluation done. {to_go} more to go!')
            to_go -= 1
        
        i += 1
    return evaluations, crashes

def save_ratio_steps_results(evaluations_array, crashes_array, no_evaluations, no_episodes_ratio_params, no_steps_params, state_space_size, this_dir=True):
    if this_dir is False:
        evaluations_file_name = 'results/4d/training_ratio_steps/ratio_steps_evaluations_array.txt'
        crashes_file_name = 'results/4d/training_ratio_steps/ratio_steps_crashes_array.txt'
        info_file_name = 'results/4d/training_ratio_steps/ratio_steps_evaluations_info.txt'
    else:
        evaluations_file_name = 'ratio_steps_evaluations_array.txt'
        crashes_file_name = 'ratio_steps_crashes_array.txt'
        info_file_name = 'ratio_steps_evaluations_info.txt'

    confirmed = input(f'About to write results to {evaluations_file_name} and info to {info_file_name}. Proceed? (y/n)') == 'y'

    if confirmed:
        np.savetxt(evaluations_file_name, evaluations_array)
        np.savetxt(crashes_file_name, crashes_array)


        lines = [f'Information on the {evaluations_file_name} file.',
             '',
             'Number of simulations used per evaluation entry: ' + str(no_evaluations),
             'Evaluated policies were trained with the following ratio of number of episodes to state space size: ' + str(no_episodes_ratio_params),
             'Evaluated policies were trained with the following numbers of policy improvement steps: ' + str(no_steps_params),
             'MDP used had the following state space size: ' + str(state_space_size)]
    
        with open(info_file_name, 'w') as f:
            for line in lines:
                f.write(line)
                f.write('\n')
        print(f'Results and info written to {evaluations_file_name} and {info_file_name}.')
    else:
        print('Did not confirm. Files NOT written.')

def ratio_steps_plot_evaluations(evaluations_array_txt_file_name, no_episodes_ratio_params, no_steps_params, surface = True, save=False):
    evaluations = np.loadtxt(evaluations_array_txt_file_name, ndmin=2)

    if surface:
        plot_x = no_episodes_ratio_params
        plot_y = no_steps_params
        X, Y = np.meshgrid(plot_x, plot_y)

        indices_x = np.arange(len(plot_x))
        indices_y = np.arange(len(plot_y))
        indices_x, indices_y = np.meshgrid(indices_x, indices_y)
        Z = evaluations[indices_x, indices_y]

        fig = plt.figure(figsize=(fsp.text_width * fsp.text_width_factor, fsp.fig_height))
        ax = fig.add_subplot(111, projection='3d')

        # plot evaluations
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='hot', linewidth=0, antialiased=False)

        # we also want to plot contour lines of constant training time (i.e., constant product of episodes per step and number of steps).
        cont_X = np.linspace(np.amin(no_episodes_ratio_params), np.amax(no_episodes_ratio_params), 30)
        cont_Y = np.linspace(np.amin(no_steps_params), np.amax(no_steps_params), 30)
        cont_X, cont_Y = np.meshgrid(cont_X, cont_Y)
        
        cont_Z = np.multiply(cont_Y, cont_X) # scalar function to view contour lines of.
        z_contour = 1.1 * np.amax(Z) # "Altitude" at which contour lines are plotted.
        ax.contour(cont_X, cont_Y, cont_Z, zdir ='z', levels = len(no_episodes_ratio_params), offset=z_contour, linewidths = 5 ,colors = '#0a97a3') # plot contours of cons
        plt.xlim(np.amin(X), np.amax(X))
        plt.ylim(np.amin(Y), np.amax(Y))
        ax.set_zlim(0, 1.1 * np.amax(Z))
        ax.set_xlabel('Ratio of number of episodes per improvement to size of state space')
        ax.set_ylabel('Number of policy improvement steps')
        ax.set_zlabel('Simulated performance')
        
        # set viewing angle
        ax.view_init(elev=26, azim = 49)
        plt.grid(True)
        plt.title('Cost function of Monte Carlo policies with varying number of episodes and improvement steps')
        plt.tight_layout()
    
    else:
        no_no_episodes_ratio_params = len(no_episodes_ratio_params)
        no_mosaic_rows = 3

        plt.figure(figsize=(fsp.text_width * fsp.text_width_factor, fsp.fig_height))

        for j in range(no_no_episodes_ratio_params):
            plt.subplot(no_mosaic_rows, int(np.ceil(no_no_episodes_ratio_params / no_mosaic_rows)), j+1)
            plt.plot(no_steps_params, evaluations[:,j], 'r-*')
            
            
            #plt.ylim(np.amin(evaluations), 0)
            plt.ylim(0, np.amax(evaluations))
            plt.grid(True)
            plt.title('Ratio number of episodes: ' + str(round(no_episodes_ratio_params[j],2)))
        plt.tight_layout()
    
    
    if save:
        plt.savefig(f'{fsp.fig_path}/4d_mc_ratio_evaluations.pdf')

def ratio_steps_plot_crash_rates(crashes_array_txt_file_name, no_evaluations, no_episodes_ratio_params, no_steps_params, surface = True, save=False):
    crashes = np.loadtxt(crashes_array_txt_file_name, ndmin=1)
    crash_rates = crashes / no_evaluations

    if surface:
        plot_x = no_episodes_ratio_params
        plot_y = no_steps_params
        X, Y = np.meshgrid(plot_x, plot_y)

        indices_x = np.arange(len(plot_x))
        indices_y = np.arange(len(plot_y))
        indices_x, indices_y = np.meshgrid(indices_x, indices_y)
        Z = crash_rates[indices_x, indices_y]

        fig = plt.figure(figsize=(fsp.text_width * fsp.text_width_factor, fsp.fig_height))
        ax = fig.add_subplot(111, projection='3d')

        # plot crash rates
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='cool', linewidth=0, antialiased=False)

        plt.grid(True)
        ax.set_xlabel('Ratio of number of episodes per improvement to size of state space')
        ax.set_ylabel('Number of policy improvement steps')
        ax.set_zlabel('Crash rate')
        ax.view_init(elev=26, azim = 49)
        plt.title('Crash rates of MC policies as function of ratio of number of episodes to state space size and number of imrpovement steps')
        plt.tight_layout()

    else:
        no_no_episodes_ratio_params = len(no_episodes_ratio_params)
        no_mosaic_rows = 3

        plt.figure(figsize=(fsp.text_width * fsp.text_width_factor, fsp.fig_height))

        for j in range(no_no_episodes_ratio_params):
            plt.subplot(no_mosaic_rows, int(np.ceil(no_no_episodes_ratio_params / no_mosaic_rows)), j+1)
            plt.plot(no_steps_params, crash_rates[:,j], 'b-*')
            
            
            #plt.ylim(np.amin(evaluations), 0)
            plt.ylim(0, np.amax(crash_rates))
            plt.grid(True)
            plt.title('Ratio number of episodes: ' + str(round(no_episodes_ratio_params[j],2)))
        plt.tight_layout()
    
    if save:
        plt.savefig(f'{fsp.fig_path}/4d_mc_ratio_crash_rates.pdf')




if __name__ == "__main__":
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
        save_epsilon_results(evaluations, crashes, bp4.epsilon_no_evaluations, bp4.epsilon_train_params, bp4.epsilon_eval_wind, this_dir=True)

    epsilon_plot = True
    if epsilon_plot:
        evaluations_file = 'results/4d/training_epsilon/epsilon_evaluations_array.txt'
        crashes_file = 'results/4d/training_epsilon/epsilon_crashes_array.txt'
        epsilon_plot_evaluations(evaluations_file, bp4.epsilon_train_params, save = True)
        epsilon_plot_crash_rates(crashes_file, bp4.epsilon_no_evaluations, bp4.epsilon_train_params, save = False)
        plt.show()
    
    ratio_steps_train = False
    if ratio_steps_train:
        ratio_steps_train_policies(bp4.epsilon_MDP, bp4.ratio_episodes_steps_ratio_params, bp4.ratio_episodes_steps_no_steps_params)

    ratio_steps_evaluate = False
    if ratio_steps_evaluate:
        evaluations, crashes = ratio_steps_evaluate_policies(bp4.epsilon_MDP, bp4.epsilon_no_evaluations, bp4.ratio_episodes_steps_ratio_params, bp4.ratio_episodes_steps_no_steps_params)
        print(f'Evaluated policies using {bp4.epsilon_no_evaluations} simulations each.')
        print(f'Evaluated policies trained with following ratios of no. episodes to state space sizes parameters: {bp4.ratio_episodes_steps_ratio_params}')
        print(f'Evaluated policies trained with following number of improvement steps: {bp4.ratio_episodes_steps_no_steps_params}')
        print('Evaluations array:')
        print(evaluations)
        print('Crashes array:')
        print(crashes)
        save_ratio_steps_results(evaluations, crashes, bp4.ratio_episodes_steps_no_evaluations, bp4.ratio_episodes_steps_ratio_params, bp4.ratio_episodes_steps_no_steps_params, bp4.epsilon_MDP.state_space.shape[0], this_dir = True)

    ratio_steps_plot = False
    if ratio_steps_plot:
        evaluations_file = 'results/4d/training_ratio_steps/ratio_steps_evaluations_array.txt'
        crashes_file = 'results/4d/training_ratio_steps/ratio_steps_crashes_array.txt'
        ratio_steps_plot_evaluations(evaluations_file, bp4.ratio_episodes_steps_ratio_params, bp4.ratio_episodes_steps_no_steps_params, surface = False, save = False)
        ratio_steps_plot_crash_rates(crashes_file, bp4.epsilon_no_evaluations, bp4.ratio_episodes_steps_ratio_params, bp4.ratio_episodes_steps_no_steps_params, surface = False, save = False)
        plt.show()