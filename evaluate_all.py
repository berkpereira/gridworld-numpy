import dynamic_programming_3d as dp3
import dynamic_programming_4d as dp4
import monte_carlo_3d as mc3
import monte_carlo_4d as mc4
import evaluate_mip_3d as emip3
import evaluate_mip_4d as emip4

from tqdm import tqdm
import time

import numpy as np
import pandas as pd
import pickle
import os

os.system('clear')

# EVALUATE BENCHMARK POLICIES AND RETURN DATAFRAME WITH RELEVANT DATA
def evaluate_rl_df(eval_MDP, eval_MDP_ID, dimension, method_str, policy, no_evaluations):
    df = pd.DataFrame(columns=['dimension', 'MDP_ID', 'wind_param', 'sol_method', 'l1_norm', 'no_solutions', 'solver_time'], index=range(no_evaluations))
    
    col_dtypes = {'dimension':'uint8', 'MDP_ID':'uint8', 'wind_param':'float16', 'sol_method':'category', 'l1_norm':'float32', 'no_solutions':'float16', 'solver_time':'float32'}

    # fill out all rows with the same value wherever relevant.
    if dimension == 4:
        df.loc[:, 'dimension'] = 4
    elif dimension == 3:
        df.loc[:, 'dimension'] = 3
    else:
        raise Exception('Invalid dimension! Must be either 3 or 4.')
    if method_str == 'MC':
        df.loc[:, 'sol_method'] = 'MC'
    elif method_str == 'DP':
        df.loc[:, 'sol_method'] = 'DP'
    else:
        raise Exception('Invalid method! Must be either DP or MC.')
    
    df.loc[:, 'MDP_ID'] = eval_MDP_ID

    if dimension == 3:
        for i in tqdm(range(no_evaluations)):
            history = mc3.generate_episode(eval_MDP, policy)
            reward = history[-1, -1]
            if reward != 0:
                l1_norm = int(1 / reward) - 1 # BEWARE OF THIS, WHICH ONLY MAKES SENSE FOR RECIPROCAL REWARD FUNCTION
                df.loc[i, 'l1_norm'] = l1_norm
            else:
                pass # leave entry as NaN
    elif dimension == 4:
        for i in tqdm(range(no_evaluations)):
            history = mc4.generate_episode(eval_MDP, policy)
            reward = history[-1, -1]
            if reward != 0:
                l1_norm = int(1 / reward) - 1 # BEWARE OF THIS, WHICH ONLY MAKES SENSE FOR RECIPROCAL REWARD FUNCTION
                df.loc[i, 'l1_norm'] = l1_norm
            else:
                pass # leave entry as NaN
    else:
        raise Exception('Invalid dimension! Must be either 3 or 4.')
    
    df = df.astype(col_dtypes)
    df.loc[:, 'wind_param'] = round(eval_MDP.direction_probability, 2)
    return df
"""
# Define metadata
grid_sizes = range(8, 9)
IDs = range(1, 5)
wind_params = np.arange(0.80, 1.02, 0.05)
no_rl_evaluations = 3000                     # NEW (number of simulations to run per solution method per benchmark MDP)
no_ip_evaluations = 300 # lower because it takes longer to simulate

# CHANGE DIMENSION HERE
dimension = 4


# Confirm to go on.
eval_confirm = input(f'Confirm evaluation with {dimension} dimensions? Type "confirm_eval" to confirm. ')
if eval_confirm == 'confirm_eval':
    print('Evaluation confirmed.')
else:
    raise Exception('Evaluation NOT confirmed!')

# One more check.
if dimension != 3 and dimension != 4:
    raise Exception('Invalid dimension! Must be either 3 or 4.')
"""



def evaluate_all(dimension, grid_sizes, IDs, wind_params, no_rl_evaluations, no_ip_evaluations):
    # Initialise dataframe for storage of each dimension's results
    no_records = len(grid_sizes) * len(IDs) * len(wind_params) * (2 * no_rl_evaluations + no_ip_evaluations)
    complete_df = pd.DataFrame(columns=['dimension', 'MDP_ID', 'wind_param', 'sol_method', 'l1_norm', 'no_solutions', 'solver_time'], index=range(no_records))

    complete_df_index = 0
    # Iterate and evaluate
    for grid_size in grid_sizes:
        for ID in IDs:
            # load policy arrays
            dp_policy_file = f"benchmark-policies/{dimension}d/dp/{grid_size}{ID}_wind_0,9_policy_array.npy"
            mc_policy_file = f"benchmark-policies/{dimension}d/mc/{grid_size}{ID}_wind_0,9_policy_array.npy"
            dp_policy_array = np.load(dp_policy_file)
            mc_policy_array = np.load(mc_policy_file)

            # convert policy arrays to actual policies (function objects)
            # this MDP is only needed in order to use array_to_policy function. Evaluation will take place with distinct MDPs below.
            if dimension == 3:
                conversion_MDP = dp3.MarkovGridWorld(grid_size=grid_size)
                dp_policy = dp3.array_to_policy(dp_policy_array, conversion_MDP) # get actual policy (function object)
                mc_policy = dp3.array_to_policy(mc_policy_array, conversion_MDP) # get actual policy (function object)
            elif dimension == 4:
                conversion_MDP = dp4.MarkovGridWorld(grid_size=grid_size)
                dp_policy = dp4.array_to_policy(dp_policy_array, conversion_MDP) # get actual policy (function object)
                mc_policy = dp4.array_to_policy(mc_policy_array, conversion_MDP) # get actual policy (function object)
            else:
                raise Exception('Invalid dimension! Must be either 3 or 4.')

            # Given policies trained at some wind parameter (hyperparameter decided before this process), we evaluate each at a sweep of wind parameters.
            # We also run the IP solution method in each case, and take note of cumulative numbers of solutions and solution times (the computation/time measures have already been carried out for RL, during training)


            for wind_param in wind_params:

                # Load benchmark MDP
                eval_MDP_file = f"benchmark-problems/{dimension}d/{grid_size}{ID}_wind_{str(round(wind_param, 2)).replace('.',',')}.p"
                with open(eval_MDP_file, 'rb') as f:
                    eval_MDP = pickle.load(f) # deserialise MDP class instance from pickled file.
                
                # EVALUATE IP
                print(f'Evaluating {grid_size}{ID}, wind {wind_param}, IP')
                if dimension == 3:
                    ip_df = emip3.evaluate_mip_df(eval_MDP, int(f'{grid_size}{ID}'), no_ip_evaluations)
                elif dimension == 4:
                    ip_df = emip4.evaluate_mip_df(eval_MDP, int(f'{grid_size}{ID}'), no_ip_evaluations)
                else:
                    raise Exception('Invalid dimension!')
                
                # EVALUATE RL
                print(f'Evaluating {grid_size}{ID}, wind {wind_param}, MC')
                mc_df = evaluate_rl_df(eval_MDP, int(f'{grid_size}{ID}'), dimension, 'MC', mc_policy, no_rl_evaluations)
                print(f'Evaluating {grid_size}{ID}, wind {wind_param}, DP')
                dp_df = evaluate_rl_df(eval_MDP, int(f'{grid_size}{ID}'), dimension, 'DP', dp_policy, no_rl_evaluations)


                all_df = pd.concat([ip_df, mc_df, dp_df], ignore_index=True)
                all_df.index = all_df.index.map(lambda x: x + complete_df_index) # shift indices to allow for the next line to work correctly
                complete_df.loc[complete_df_index:(complete_df_index + all_df.shape[0]) - 1] = all_df # NEED the -1 because these limits are INCLUSIVE, which is unusual
                complete_df_index += all_df.shape[0]

    print('Results dataframe:')
    print(complete_df)
    print('Analysis metadata:')
    print(f'Dimension: {dimension}')
    print(f'Grid sizes: {list(grid_sizes)}')
    print(f'IDs: {list(IDs)}')
    print(f'Wind parameters: {list(wind_params)}')
    print(f'Number of RL evaluations: {no_rl_evaluations}')
    print(f'Number of IP evaluations: {no_ip_evaluations}')


    # HERE WE SAVE IT
    results_file_name = f'{dimension}d-grids-{min(list(grid_sizes))}-{max(list(grid_sizes))}-IDs-{min(list(IDs))}-{max(list(IDs))}-winds-{round(min(list(wind_params)), 2)}-{round(max(list(wind_params)), 2)}-noRL-{no_rl_evaluations}-noIP-{no_ip_evaluations}.csv'
    
    complete_df.to_csv(results_file_name, index=False)


"""
grid_sizes_list = [range(4,5), range(5,6),
                   range(6,7), range(7,8),
                   range(8,9), range(9,10),
                   range(10,11), range(11,12),
                   range(12,13), range(13,14),]
"""

grid_sizes_list = [range(6,7), range(7,8),
                   range(8,9), range(9,10),
                   range(10,11), range(11,12),
                   range(12,13), range(13,14),]

IDs = range(1,5)
wind_params = np.arange(0.80, 1.02, 0.05)
no_rl_evaluations = 3000                     # NEW (number of simulations to run per solution method per benchmark MDP)
no_ip_evaluations = 300 # lower because it takes longer to simulate

dimension = 3


os.system('say evaluation begun')
st = time.time()
for grid_size_range in grid_sizes_list:
    # evaluate and write files, one file per grid size.
    evaluate_all(dimension=dimension, grid_sizes=grid_size_range, IDs=IDs, wind_params=wind_params, no_rl_evaluations=no_rl_evaluations, no_ip_evaluations=no_ip_evaluations)
et = time.time()


os.system('say evaluation finished')
os.system(f'say evaluation took {et - st} seconds')
print(f'Time for this analysis: {et - st} seconds')
