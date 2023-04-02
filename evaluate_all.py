import dynamic_programming_3d as dp3
import dynamic_programming_4d as dp4
import monte_carlo_3d as mc3
import monte_carlo_4d as mc4
import evaluate_mip_3d as emip3
import evaluate_mip_4d as emip4

import numpy as np
import pandas as pd
import pickle
import os

# EVALUATE BENCHMARK POLICIES AND RETURN DATAFRAME WITH RELEVANT DATA
def evaluate_rl_df(eval_MDP, eval_MDP_ID, dimension, method_str, policy, no_evaluations):
    df = pd.DataFrame(columns=['dimension', 'MDP_ID', 'sol_method', 'l1_norm', 'no_solutions', 'solver_time'], index=range(no_evaluations))
    
    col_dtypes = {'dimension':'uint8', 'MDP_ID':'uint8', 'sol_method':'category', 'l1_norm':'float32', 'no_solutions':'float16', 'solver_time':'float32'}

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
        for i in range(no_evaluations):
            history = mc3.generate_episode(eval_MDP, policy)
            reward = history[-1, -1]
            l1_norm = int(1 / reward) - 1 # BEWARE OF THIS, WHICH ONLY MAKES SENSE FOR RECIPROCAL REWARD FUNCTION
            df.loc[i, 'l1_norm'] = l1_norm
    elif dimension == 4:
        for i in range(no_evaluations):
            history = mc4.generate_episode(eval_MDP, policy)
            reward = history[-1, -1]
            l1_norm = int(1 / reward) - 1 # BEWARE OF THIS, WHICH ONLY MAKES SENSE FOR RECIPROCAL REWARD FUNCTION
            df.loc[i, 'l1_norm'] = l1_norm
    else:
        raise Exception('Invalid dimension! Must be either 3 or 4.')
    
    df = df.astype(col_dtypes)
    return df

# Define metadata
grid_sizes = range(4, 14)
IDs = range(1, 5)
wind_params = np.arange(0.70, 1.02, 0.05)
no_rl_evaluations = 5000                     # NEW (number of simulations to run per solution method per benchmark MDP)
no_ip_evaluations = 500 # lower because it takes longer to train

# CHANGE DIMENSION HERE
dimension = 'bogus'

# Confirm to go on.
eval_confirm = input(f'Confirm evaluation with {dimension} dimensions? Type "confirm_eval" to confirm. ')
if eval_confirm == "confirm_eval":
    print('Evaluation confirmed.')
else:
    raise Exception('Evaluation NOT confirmed!')

# One more check.
if dimension != 3 and dimension != 4:
    raise Exception('Invalid dimension! Must be either 3 or 4.')




# Initialise dataframe for storage of each dimension's results
no_records = len(grid_sizes) * len(IDs) * len(wind_params) * (2 * no_rl_evaluations + no_ip_evaluations)
complete_df = pd.DataFrame(columns=['dimension', 'MDP_ID', 'sol_method', 'l1_norm', 'no_solutions', 'solver_time'], index=range(no_records))


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
            eval_MDP_file = f"benchmark_problems/{dimension}d/{grid_size}{ID}_wind_{str(round(wind_param, 2)).replace('.',',')}.p"
            with open(eval_MDP_file, 'rb') as f:
                eval_MDP = pickle.load(f) # deserialise MDP class instance from pickled file.
            
            # EVALUATE IP
            if dimension == 3:
                ip_df = emip3.evaluate_mip_df(eval_MDP, int(f'{grid_size}{ID}'), no_ip_evaluations)
            elif dimension == 4:
                ip_df = emip4.evaluate_mip_df(eval_MDP, int(f'{grid_size}{ID}'), no_ip_evaluations)
            else:
                raise Exception('Invalid dimension!')
            
            # EVALUATE RL
            mc_df = evaluate_rl_df(eval_MDP, int(f'{grid_size}{ID}'), dimension, 'MC', mc_policy, no_rl_evaluations)
            dp_df = evaluate_rl_df(eval_MDP, int(f'{grid_size}{ID}'), dimension, 'DP', dp_policy, no_rl_evaluations)
