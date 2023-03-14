import numpy as np
import os

import dynamic_programming_4d as dp4
import solve_mip4d as mip4
import monte_carlo_4d as mc4

# this will take an MDP problem, solve it via MIP, simulate time steps by sampling MDP dynamics, recompute MIP solutions as needed if the real outcomes
# deviate from the expected ones at any point.
# ATTENTION: initial state is just a 2-element array, as expected in the MIP functions!
def mip_simulate_closed_loop(sim_MDP, sim_mip_initial_state, sim_initial_velocity_index):
    MDP = sim_MDP
    
    initial_state = sim_mip_initial_state
    initial_velocity_index = sim_initial_velocity_index
    
    # Will try initialising ampl object just once and from then on just rewriting parameter values and re-solving problem.
    # IF THAT DOESN'T WORK, then I will reinitialise ampl object every time the problem has to be re-solved. This will make process take longer,
    # but it's okay because we still only extract the elapsed time in solving the problem!
    ampl = mip4.initialise_ampl()

    # we will have to keep recomputing MIP solutions as we simulate an episode.
    # each MIP solution entails an input MDP with a certain "starting" altitude in the attribute MDP.max_altitude.
    # As long as we are looking at MDPs with max_altitude above 0 (the only valid sort of MDP), all good.
    # As soon as it hits zero (or negative), the simulation has ended and we end it there.
    mip_history, actions, solve_time = mip4.mip_history_and_actions_from_mdp(MDP, initial_state, initial_velocity_index, ampl)

    sim_compute_time = solve_time # this will keep accumulating
    sim_mip_solutions = 1 # this will keep accumulating
    sim_history = np.zeros(shape=(sim_MDP.max_altitude + 1, 5))
    
    sim_history_index = 0
    while MDP.max_altitude > 0:

        # if solution failed!
        # this can be because the experienced state was a crash (obstacle) or because the agent ended up at a state from which
        # the MIP problem was insoluble! if problem was insoluble, the episode is just kind of cut-off suddenly. This will always happen NEXT TO
        # EITHER AN OBSTACLE OR A BOUNDARY!
        if mip_history is False:
            try: # this will fail if the MIP could not be solved to begin with
                sim_history[sim_history_index, :4] = experienced_next_state
                sim_history = sim_history[:(sim_history_index + 1)]
                break
            
            # once more, in case of error, return same number of argument as in successful case.
            # this makes it easier to unpack as usual
            except:
                return False, False, False


        step = 0
        # first in this sequence will always match, since it's the initial state
        sim_history[sim_history_index] = mip_history[step]
        sim_history_index += 1
        # As long as experience is matching what's expected from MIP solution!
        experienced_next_state = MDP.state_transition(mip_history[step,:4], actions[step])
        while (step + 1) < mip_history.shape[0] and np.array_equal(experienced_next_state, mip_history[step+1,:4]):
            sim_history[sim_history_index] = mip_history[step + 1]
            step += 1
            sim_history_index += 1
            if (step + 1) < mip_history.shape[0]:
                experienced_next_state = MDP.state_transition(mip_history[step,:4], actions[step])


        # if we've already filled out the entire simulated history down to the ground, end here.
        if sim_history_index >= sim_history.shape[0]:
            break

        # final step.
        # assign experienced state, since we've landed now.
        if sim_history_index == sim_history.shape[0] - 1:
            sim_history[sim_history_index, :4] = experienced_next_state
        # if solution hasn't failed (likely due to obstacle crash) and we haven't finished the simulation yet, keep going...
        max_altitude = int(experienced_next_state[0])

        # take in initial state for next solution in MIP form (just 2-element array!)
        mip_next_state = experienced_next_state[2:4]
        MDP = dp4.MarkovGridWorld(grid_size=sim_MDP.grid_size, direction_probability=sim_MDP.direction_probability, obstacles=sim_MDP.obstacles, landing_zone=sim_MDP.landing_zone, max_altitude=max_altitude)
        mip_history, actions, solve_time = mip4.mip_history_and_actions_from_mdp(MDP, mip_next_state, experienced_next_state[1], ampl)

        # accrue compute time and number of computed MIP solutions
        sim_compute_time += solve_time
        sim_mip_solutions += 1

    return sim_history, sim_mip_solutions, sim_compute_time

# STILL HAVE TO ITERATE THIS OVER MULTIPLE TIMES, OBVIOUSLY
# AND THEN RETURN THE AVERAGE "SCORE"
def evaluate_mip(eval_MDP, no_evaluations):
    
    # we first need to generate a random initial state that isn't infeasible straight away,
    # e.g., a boundary state heading into the boundary, stuff which the IP would have issues computing to begin with
    sim_history = False
    while sim_history is False:
        sim_initial_velocity_index = np.random.randint(0,4)
        sim_mip_initial_state = np.array([np.random.randint(0, eval_MDP.grid_size), np.random.randint(0, eval_MDP.grid_size)], dtype='int32')
        sim_history, sim_mip_solutions, sim_compute_time = mip_simulate_closed_loop(sim_MDP=eval_MDP, sim_mip_initial_state=sim_mip_initial_state, sim_initial_velocity_index=sim_initial_velocity_index)

    # did not get to land, for whatever reason
    # need to check whether it:
    # CASE 1: hit an obstacle or was bound to and hence couldn't solve the MIP from there.
    # CASE 2: it just found itself going against a boundary and hence couldn't solve MIP from there.
    
    # CASE 1 must be negatively reflected in the average score, consistently with the way we come to do it with RL
    # CASE 2 must not. not sure how to handle these cases, but perhaps, at least for reasonably sized problems, it's best to
    # just pretend they didn't happen and keep simulating from there.
    if sim_history[0,0] > 0:
        pass

    # when we break out of the while condition it means we've solved a problem that wasn't infeasible to begin with!
    return sim_history, sim_mip_solutions, sim_compute_time, score




if __name__ == "__main__":
    os.system('clear')
    #MDP_list = [dp4.MarkovGridWorld(grid_size=5, direction_probability=1, obstacles=np.array([[]]), landing_zone=np.array([0,0]), max_altitude=10),
    #            dp4.MarkovGridWorld(grid_size=4, direction_probability=1, obstacles=np.array([[]]), landing_zone=np.array([0,0]), max_altitude=5)]
    
    #test_MDP = dp4.MarkovGridWorld(grid_size=6, direction_probability=0.90, obstacles=np.array([[0,1], [10,0], [4,21], [13,6], [20,20]]), landing_zone=np.array([3,3]), max_altitude=30)
    MDP = dp4.MarkovGridWorld(grid_size=4, direction_probability=0.5,obstacles=np.array([[0,0]]), landing_zone = np.array([1,1]), max_altitude = 12)
    #sim_history, sim_mip_solutions, sim_compute_time = mip_simulate_closed_loop(sim_MDP=MDP, sim_mip_initial_state=np.array([2,1]), sim_initial_velocity_index=2)
    for i in range(3):
        sim_history = evaluate_mip(eval_MDP=MDP, no_evaluations=0)
        print(sim_history)
        print()
        print()
        mc4.play_episode(MDP, None, sim_history)
    #print(f"Number of MIP solutions: {sim_mip_solutions}.\nCumulative time spent computing MIP solutions: {sim_compute_time} seconds.")
    #mc4.play_episode(test_MDP, None, sim_history)