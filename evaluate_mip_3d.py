"""
import numpy as np
import os

import dynamic_programming_3d as dp3
import solve_mip3d as mip3
import monte_carlo_3d as mc3

# this will take an MDP problem, solve it via MIP, simulate time steps by sampling MDP dynamics, recompute MIP solutions as needed if the real outcomes
# deviate from the expected ones at any point.
# initial state is just a 2-element array, as expected in the MIP functions!
def mip_simulate_closed_loop(sim_MDP, sim_mip_initial_state, sim_initial_velocity_index):
    MDP = sim_MDP
    
    initial_state = sim_mip_initial_state
    initial_velocity_index = sim_initial_velocity_index
    
    # Will try initialising ampl object just once and from then on just rewriting parameter values and re-solving problem.
    # IF THAT DOESN'T WORK, then I will reinitialise ampl object every time the problem has to be re-solved. This will make process take longer,
    # but it's okay because we still only extract the elapsed time in solving the problem!
    ampl = mip3.initialise_ampl()

    # we will have to keep recomputing MIP solutions as we simulate an episode.
    # each MIP solution entails an input MDP with a certain "starting" altitude in the attribute MDP.max_altitude.
    # As long as we are looking at MDPs with max_altitude above 0 (the only valid sort of MDP), all good.
    # As soon as it hits zero (or negative), the simulation has ended and we end it there.
    mip_history, actions, solve_time = mip3.mip_history_and_actions_from_mdp(MDP, initial_state, ampl)

    sim_compute_time = solve_time # this will keep accumulating
    sim_mip_solutions = 1 # this will keep accumulating
    sim_history = np.zeros(shape=(sim_MDP.max_altitude + 1, 4))
    
    sim_history_index = 0
    while MDP.max_altitude > 0:

        # if solution failed!
        # this can be because the experienced state was a crash (obstacle) or because the agent ended up at a state from which
        # the MIP problem was insoluble! if problem was insoluble, the episode is just kind of cut-off suddenly. This will always happen NEXT TO
        # EITHER AN OBSTACLE OR A BOUNDARY!
        if mip_history is False:
            sim_history[sim_history_index, :3] = experienced_next_state
            sim_history = sim_history[:(sim_history_index + 1)]
            break

        step = 0
        # first in this sequence will always match, since it's the initial state
        sim_history[sim_history_index] = mip_history[step]
        sim_history_index += 1
        # As long as experience is matching what's expected from MIP solution!
        experienced_next_state = MDP.state_transition(mip_history[step,:3], actions[step])
        while (step + 1) < mip_history.shape[0] and np.array_equal(experienced_next_state, mip_history[step+1,:3]):
            sim_history[sim_history_index] = mip_history[step + 1]
            step += 1
            sim_history_index += 1
            if (step + 1) < mip_history.shape[0]:
                experienced_next_state = MDP.state_transition(mip_history[step,:3], actions[step])

        #print('broke while condition')
        # if we've already filled out the entire simulated history down to the ground, end here.
        if sim_history_index >= sim_history.shape[0]:
            break

        # final step.
        # assign experienced state, since we've landed now.
        if sim_history_index == sim_history.shape[0] - 1:
            sim_history[sim_history_index, :3] = experienced_next_state
        # if solution hasn't failed (likely due to obstacle crash) and we haven't finished the simulation yet, keep going...
        max_altitude = int(experienced_next_state[0])

        # take in initial state for next solution in MIP form (just 2-element array!)
        mip_next_state = experienced_next_state[1:3]
        MDP = dp3.MarkovGridWorld(grid_size=sim_MDP.grid_size, direction_probability=sim_MDP.direction_probability, obstacles=sim_MDP.obstacles, landing_zone=sim_MDP.landing_zone, max_altitude=max_altitude)
        mip_history, actions, solve_time = mip3.mip_history_and_actions_from_mdp(MDP, mip_next_state, ampl)

        # accrue compute time and number of computed MIP solutions
        sim_compute_time += solve_time
        sim_mip_solutions += 1

    return sim_history, sim_mip_solutions, sim_compute_time




if __name__ == "__main__":
    os.system('clear')
    #MDP_list = [dp3.MarkovGridWorld(grid_size=5, direction_probability=1, obstacles=np.array([[]]), landing_zone=np.array([0,0]), max_altitude=10),
    #            dp3.MarkovGridWorld(grid_size=4, direction_probability=1, obstacles=np.array([[]]), landing_zone=np.array([0,0]), max_altitude=5)]
    
    test_MDP = dp3.MarkovGridWorld(grid_size=25, direction_probability=1, obstacles=np.array([[0,1], [10,0], [4,21], [13,6], [20,20]]), landing_zone=np.array([15,15]), max_altitude=50)
    sim_history, sim_mip_solutions, sim_compute_time = mip_simulate_closed_loop(sim_MDP=test_MDP, sim_mip_initial_state=np.array([5,5]), sim_initial_velocity_index=0)
    print(sim_history)
    print()
    print(f"Number of MIP solutions: {sim_mip_solutions}.\nCumulative time spent computing MIP solutions: {sim_compute_time} seconds.")
    mc3.play_episode(test_MDP, None, sim_history)
"""

import numpy as np
import os

import dynamic_programming_3d as dp3
import solve_mip3d as mip3
import monte_carlo_3d as mc3

# this will take an MDP problem, solve it via MIP, simulate time steps by sampling MDP dynamics, recompute MIP solutions as needed if the real outcomes
# deviate from the expected ones at any point.
# ATTENTION: initial state is just a 2-element array, as expected in the MIP functions!
def mip_simulate_closed_loop(sim_MDP, sim_mip_initial_state):
    MDP = sim_MDP
    
    initial_state = sim_mip_initial_state
    
    # Will try initialising ampl object just once and from then on just rewriting parameter values and re-solving problem.
    # IF THAT DOESN'T WORK, then I will reinitialise ampl object every time the problem has to be re-solved. This will make process take longer,
    # but it's okay because we still only extract the elapsed time in solving the problem!
    ampl = mip3.initialise_ampl()

    # we will have to keep recomputing MIP solutions as we simulate an episode.
    # each MIP solution entails an input MDP with a certain "starting" altitude in the attribute MDP.max_altitude.
    # As long as we are looking at MDPs with max_altitude above 0 (the only valid sort of MDP), all good.
    # As soon as it hits zero (or negative), the simulation has ended and we end it there.
    mip_history, actions, solve_time = mip3.mip_history_and_actions_from_mdp(MDP, initial_state, ampl)

    sim_compute_time = solve_time # this will keep accumulating
    sim_mip_solutions = 1 # this will keep accumulating
    sim_history = np.zeros(shape=(sim_MDP.max_altitude + 1, 4))
    
    sim_history_index = 0
    while MDP.max_altitude > 0:

        # if solution failed!
        # this can be because the experienced state was a crash (obstacle) or because the agent ended up at a state from which
        # the MIP problem was insoluble! if problem was insoluble, the episode is just kind of cut-off suddenly. This will always happen NEXT TO
        # EITHER AN OBSTACLE OR A BOUNDARY!
        if mip_history is False:
            try: # this will FAIL if the MIP could not be solved to begin with, otherwise something's gone off track mid-simulation
                sim_history[sim_history_index, :3] = experienced_next_state
                sim_history = sim_history[:(sim_history_index + 1)]
                break
            
            # once more, in case of error, return same number of argument as in successful case.
            # this makes it easier to unpack as usual.
            # this except is triggered if the MIP was insoluble from the start.
            except:
                return False, False, False


        step = 0
        # first in this sequence will always match, since it's the initial state
        sim_history[sim_history_index] = mip_history[step]
        sim_history_index += 1
        # As long as experience is matching what's expected from MIP solution!
        experienced_next_state = MDP.state_transition(mip_history[step,:3], actions[step])
        while (step + 1) < mip_history.shape[0] and np.array_equal(experienced_next_state, mip_history[step+1,:3]):
            sim_history[sim_history_index] = mip_history[step + 1]
            step += 1
            sim_history_index += 1
            if (step + 1) < mip_history.shape[0]:
                experienced_next_state = MDP.state_transition(mip_history[step,:3], actions[step])


        # if we've already filled out the entire simulated history down to the ground, end here.
        if sim_history_index >= sim_history.shape[0]:
            break

        # final step.
        # assign experienced state, since we've landed now.
        if sim_history_index == sim_history.shape[0] - 1:
            sim_history[sim_history_index, :3] = experienced_next_state
        # if solution hasn't failed (likely due to obstacle crash) and we haven't finished the simulation yet, keep going...
        max_altitude = int(experienced_next_state[0])

        # take in initial state for next solution in MIP form (just 2-element array!)
        mip_next_state = experienced_next_state[1:3]
        MDP = dp3.MarkovGridWorld(grid_size=sim_MDP.grid_size, direction_probability=sim_MDP.direction_probability, obstacles=sim_MDP.obstacles, landing_zone=sim_MDP.landing_zone, max_altitude=max_altitude)
        mip_history, actions, solve_time = mip3.mip_history_and_actions_from_mdp(MDP, mip_next_state, ampl)

        # accrue compute time and number of computed MIP solutions
        sim_compute_time += solve_time
        sim_mip_solutions += 1

    # ended prematurely either due to:
    # experienced crash,
    # imminent crash, or
    # boundary error.
    """
    if MDP.max_altitude > 0 and sim_history[-1,0] > 0:
        last_recorded_step = sim_history[-1]
        direction_array = MDP.action_to_direction[last_recorded_step[1]][0]
        imminent_2d_state = np.clip(last_recorded_step[1:3] + direction_array, 0, MDP.grid_size - 1)
        for obstacle in MDP.obstacles:
            if np.array_equal(imminent_2d_state, obstacle) and not np.array_equal(last_recorded_step[1:3], obstacle): # imminent crash that has NOT happened yet
                imminent_step = np.array([MDP.max_altitude-1, last_recorded_step[1], imminent_2d_state[0], imminent_2d_state[1], 0])
                sim_history = np.append(sim_history, np.array(imminent_step, ndmin=2), axis=0) # add on the imminent crashed state to the history
                break
        else: # i.e., did NOT find an obstacle imminently to be crashed into!
            # at this point we will just let the history be and cut off at the boundary error. We'll check these cases in the evaluation function
            pass
    """
        

    return sim_history, sim_mip_solutions, sim_compute_time

# STILL HAVE TO ITERATE THIS OVER MULTIPLE TIMES, OBVIOUSLY
# AND THEN RETURN THE AVERAGE "SCORE"
def evaluate_mip(eval_MDP, no_evaluations):
    
    cumulative_score = 0
    evaluation_no = 0
    crashes = 0
    landed_solve_time = 0
    landed_solve_no = 0
    while evaluation_no < no_evaluations:
        crashed = False
        # we need to generate a random initial state that isn't infeasible straight away,
        # e.g., a boundary state heading into the boundary, stuff which the IP would have issues computing to begin with.
        # we do this at the same time as we solve the full problem.
        sim_history = False
        while sim_history is False:
            sim_mip_initial_state = np.array([np.random.randint(0, eval_MDP.grid_size), np.random.randint(0, eval_MDP.grid_size)], dtype='int32')
            sim_history, sim_mip_solutions, sim_compute_time = mip_simulate_closed_loop(sim_MDP=eval_MDP, sim_mip_initial_state=sim_mip_initial_state)

        
        
        
        mc3.play_episode(eval_MDP, None, sim_history)




        # check if agent crashed
        for obstacle in eval_MDP.obstacles:
            if np.array_equal(sim_history[-1,1:3], obstacle): # crashed
                # no score added
                crashes += 1
                evaluation_no += 1
                crashed = True
                break
        
        # agent crashed, it's taken care of, go on to the next one.
        if crashed:
            continue

        """
        # problem stopped early without a crash --> boundary problem.
        # we will IGNORE these cases.
        if sim_history[-1,0] > 0 and not crashed:
            # do NOT increment score NOR evaluation_no
            continue
        """

        # we are finally left with cases where agent DID land
        cumulative_score += eval_MDP.reward(sim_history[-1,:3])
        evaluation_no += 1
        landed_solve_time += sim_compute_time
        landed_solve_no += sim_mip_solutions


    average_landed_return = cumulative_score / (no_evaluations - crashes)
    average_landed_solve_time = landed_solve_time / (no_evaluations - crashes)
    average_landed_solve_no = landed_solve_no / (no_evaluations - crashes)
    return average_landed_return, average_landed_solve_time, average_landed_solve_no, crashes




if __name__ == "__main__":
    os.system('clear')
    MDP = dp3.MarkovGridWorld(grid_size=4, direction_probability=0.80,obstacles=np.array([[0,0], [2,2]]), landing_zone = np.array([1,1]), max_altitude = 5)
    no_evaluations = 10
    avg_landed_return, avg_landed_solve_time, avg_landed_solve_no, crashes = evaluate_mip(MDP, no_evaluations)
    print()
    print()
    print(f'Number of MIP evaluations (excluding boundary problems): {no_evaluations}')
    print(f'Number of simulated crashes: {crashes}')
    print(f'Simulation crash rate: {crashes/no_evaluations * 100}%')
    print(f'Average score in NON-CRASHED simulations: {avg_landed_return}')
    print(f'Average solve time in NON-CRASHED simulations: {avg_landed_return} seconds')
    print(f'Average no. of solutions in NON-CRASHED simulations: {avg_landed_solve_no}')
    print()
    print()
