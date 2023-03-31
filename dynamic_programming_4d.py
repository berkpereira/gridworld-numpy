import pickle
import os
import time
import numpy as np
from scipy.spatial.distance import cityblock
from scipy.stats import norm

# for generality we're defining the policy as being a function of an action and a current state
def random_walk(action, state):
    return 1/3

def straight_walk(action, state):
    if action == 0:
        return 1
    else:
        return 0

# this class codifies all the dynamics of the problem: a simple gridworld
# as a starter in implementing GridWorld stochastics, I will try to program simple stochastics into this gridworld's dynamics
# E.G., given a certain action in a rectangular direction, we can assign the successor state some stochastics like
# 85% probability of ending up where you expected, and 5% in each of the 3 other directions.
# the argument direction_probability controls the probability with which we can expect the agent's selected action to result in the 'expected' successor state.
class MarkovGridWorld():
    def __init__(self, grid_size=3, discount_factor=1, direction_probability = 1, obstacles = np.array([0,0], ndmin=2, dtype='int32'), landing_zone = np.array([1, 1], dtype='int32'), max_altitude = None):
        self.grid_size = grid_size
        self.crash_penalty = -10000

        # self.landing_zone given as a 2-element row vector
        self.landing_zone = landing_zone

        # obstacles
        self.obstacles = obstacles
        
        # set default max_altitude to 2 times grid size.
        # this is the minimum altitude that allows agent to get from one corner to the other.
        if max_altitude is None:
            self.max_altitude = self.grid_size * 2
        else:
            self.max_altitude = max_altitude
        
        # 0: forward
        # 1: right
        # 2: left
        self.action_space = (0, 1, 2)

        # discount factor
        self.discount_factor = discount_factor

        # define normal distribution of reward.
        # using a standard normal pdf
        self.reward_dist = norm()
        
        # terminal state is assigned directly to a crash or right after landing manoeuvre reward has been collected, subsequent rewards are always 0.
        # just a reserved state for representation of episode termination in dynamic programming algorithms.
        self.terminal_state = np.array([-1, -1, -1, -1], dtype='int32')

        # index with heading, THEN index with action.
        self.action_to_direction = {
            0: {0: np.array([1, 0], dtype='int32'),  1: np.array([0, -1], dtype='int32'), 2: np.array([0, 1], dtype='int32')}, # heading down
            1: {0: np.array([0, 1], dtype='int32'),  1: np.array([1, 0], dtype='int32'), 2: np.array([-1, 0], dtype='int32')}, # heading right
            2: {0: np.array([-1, 0], dtype='int32'), 1: np.array([0, 1], dtype='int32'), 2: np.array([0, -1], dtype='int32')}, # heading up
            3: {0: np.array([0, -1], dtype='int32'), 1: np.array([-1, 0], dtype='int32'), 2: np.array([1, 0], dtype='int32')}, # heading left
        }

        


        self.rng = np.random.default_rng() # construct a default numpy random number Generator class instance, to use in stochastics
        self.direction_probability = direction_probability
        self.prob_other_directions = (1 - self.direction_probability) / 2 # now divide by 2 for other directions
        
        # for ease of iterating over all states, define an explicit list of all states. 
        # len(self.action_to_direction gives number of valid headings.
        self.state_space = np.zeros(shape=(self.grid_size**2 * (self.max_altitude + 2) * len(self.action_to_direction),4), dtype='int32')

        # 4D array shape to use for value functions, etc.
        self.problem_shape = (self.max_altitude + 2, len(self.action_to_direction), self.grid_size, self.grid_size)
        
        state_counter = 0

        for altitude in range(self.max_altitude, -2, -1):
            for heading in range(len(self.action_to_direction)): # headings from 0 to 3
                for row in range(self.grid_size):
                    for col in range(self.grid_size):
                        # 0 < altitude <= self.max_altitude means in-flight
                        # altitude > self.max_altitude signifies landing manoeuvre has been pulled from an altitude of altitude % self.max_altitude:
                        # E.g., if self.max_altitude is 3 and a state has 3rd coordinate of 4, this signifies that agent has landed from an altitude of 1 (ideal landing).
                        # 0 altitude is terminal/crash, reserved for self.terminal_state
                        self.state_space[state_counter, :] = [altitude, heading, row, col]
                        state_counter += 1

    # this method is now extended to 4D.
    # this method is used within the context of potential successors, so we don't have to worry about all the weird possible
    # state differences, only those which are allowed by the MDP dynamics.
    def state_difference_to_action(self, difference, start_state):
        
        # the below has become redundant because tre dynamics force the agent to turn at corners anyway.
        # but we will leave it in so that it's clear that the agent "learns" to do it nonetheless.
        if np.array_equal(difference[2:], np.array([0,0])): # catch cases on the boundary of the grid
            if difference[1] == 0: # same heading
                return 0 # keep going "forward"
            elif difference[1] == -1 or difference[1] == 3: # at corner, heading difference showing it's a right turn
                return 1
            else:
                return 2 # at a corner, heading difference showing it's a left turn
        else:
            for action in range(len(self.action_space)):
                if np.array_equal(difference[2:], self.action_to_direction[start_state[1]][action]):
                    return action
        
        # if nothing is found, terminal states and so on
        return 2

    def direction_to_heading(self, direction):
        if np.array_equal(direction, [1,0]):
            return 0
        if np.array_equal(direction, [0,1]):
            return 1
        if np.array_equal(direction, [-1,0]):
            return 2
        if np.array_equal(direction, [0,-1]):
            return 3

    # must redefine the reward to be spread out on the ground.
    # however, prevent any reward from being there at the obstacles! can't crash into a building at ground level and expect to get any reward!
    """
    def reward(self, state):
        if state[0] == 0:
            for obstacle in self.obstacles:
                if np.array_equal(state[2:], obstacle):
                    return 0
            manhattan_distance = cityblock(state[2:], self.landing_zone)
            return - manhattan_distance
            # return norm.pdf(manhattan_distance) / norm.pdf(0) # normalise against the max reward available
        else:
            return 0
    """

    """
    def reward(self, state):
        for obstacle in self.obstacles:
            if np.array_equal(state[2:], obstacle):
                    return -2 * 1000 * self.grid_size
        
        # landed without crash, because we've checked
        if state[0] == 0:
            manhattan_distance = cityblock(state[2:], self.landing_zone)
            return - manhattan_distance
        
        return 0
    """

    # yet another possible reward function
    def reward(self, state):
        if state[0] == 0:
            for obstacle in self.obstacles:
                if np.array_equal(state[2:], obstacle):
                    return 0
            else:
                manhattan_distance = cityblock(state[2:], self.landing_zone)
                return 1 / (manhattan_distance + 1)
        return 0

    """
    def reward(self, state):
        for obstacle in self.obstacles:
            if np.array_equal(state[2:], obstacle):
                return self.crash_penalty # very negative number
        
        if state[0] == 0: # and not crashed, because we've checked
            manhattan_distance = cityblock(state[2:], self.landing_zone)
            return -manhattan_distance
        else:
            return 0
    """


    # returns the probability of a successor state, given a current state and an action.
    # crucial to define these in this generalised form, in order to implement general policy evaluation algorithm.
    def environment_dynamics(self, successor_state, current_state, action):
        if current_state[0] <= 0: # agent has landed already
            if np.array_equal(successor_state, self.terminal_state):
                return 1 # can only be taken to terminal_state
            else:
                return 0 # anywhere else is impossible to succeed the current_state, given the condition above.
        
        # Also consider obstacles:
        # it's not very realistic nor practical to just get rid of an obstacle's grid from the state space.
        # the best option is to treat it like another crashed state.
        # this way, it's still there, and it's possible for a bad policy or for environment dynamics stochasticity to
        # make the agent crash against it --> most realistic.
        """
        for obstacle in self.obstacles:
            if np.array_equal(current_state[2:], obstacle):
                if np.array_equal(successor_state, self.terminal_state):
                    return 1 # can only be taken to terminal_state
                else:
                    return 0 # anywhere else is impossible to succeed the current_state, given the condition above.
        """
        for obstacle in self.obstacles:
            if np.array_equal(current_state[2:], obstacle):
                if np.array_equal(successor_state, current_state):
                    return 1 # can only get stuck in itself and reap huge penalties
                else:
                    return 0 # anywhere else is impossible to succeed the current_state, given the condition above.


        # consider BOUNDARIES OF DOMAIN. agent was exploting these to just descend vertically, so we need to prevent it from being
        # able to do that. We must force the agent to turn it when it reaches a domain boundary.
        # Give it the choice of which way to turn. The wind stochasticity gets implemented differently, because it doesn't make sense to
        # make it possible for the wind to make the agent go forward (ends up with vertical descent again).
        # Instead we will let the agent turn as intended with probability self.direction_probability, and go the other way
        # with probability (1 - self.direction_probability).
        
        # first consider the CORNERS.
        # if the agent is at a corner, it is NECESSARILY heading towards it, so it must be forced to turn, regardless of its action.
        if (current_state[2] == 0 and current_state[3] == 0): # top-left corner
            if current_state[1] == 2: # heading up
                if np.array_equal(successor_state, np.array([current_state[0] - 1, 1, 0, 1], dtype='int32')): # turn to global right
                    return 1
                else:
                    return 0
            else: # heading left
                if np.array_equal(successor_state, np.array([current_state[0] - 1, 0, 1, 0], dtype='int32')): # turn to global down
                    return 1
                else:
                    return 0
        if (current_state[2] == 0 and current_state[3] == (self.grid_size - 1)): # top-right corner
            if current_state[1] == 1: # heading right
                if np.array_equal(successor_state, np.array([current_state[0] - 1, 0, 1, self.grid_size - 1], dtype='int32')): # turn to global down
                    return 1
                else:
                    return 0
            else: # heading up
                if np.array_equal(successor_state, np.array([current_state[0] - 1, 3, 0, self.grid_size - 2], dtype='int32')): # turn to global left
                    return 1
                else:
                    return 0
        if (current_state[2] == (self.grid_size - 1) and current_state[3] == 0): # bottom-left corner
            if current_state[1] == 0: # heading down
                if np.array_equal(successor_state, np.array([current_state[0] - 1, 1, self.grid_size - 1, 1], dtype='int32')): # turn to global right
                    return 1
                else:
                    return 0
            else: # heading left
                if np.array_equal(successor_state, np.array([current_state[0] - 1, 2, self.grid_size - 2, 0], dtype='int32')): # turn to global up
                    return 1
                else:
                    return 0
        if (current_state[2] == (self.grid_size - 1) and current_state[3] == (self.grid_size - 1)): # bottom-right corner
            if current_state[1] == 0: # heading down
                if np.array_equal(successor_state, np.array([current_state[0] - 1, 3, self.grid_size - 1, self.grid_size - 2], dtype='int32')): # turn to global left
                    return 1
                else:
                    return 0
            else: # heading right
                if np.array_equal(successor_state, np.array([current_state[0] - 1, 2, self.grid_size - 2, self.grid_size - 1], dtype='int32')): # turn to global up
                    return 1
                else:
                    return 0


        # Now consider NON-CORNER BOUNDARIES
        if current_state[2] == 0: # top boundary
            if current_state[1] == 2: # heading up, must be forced to turn
                if action == 0: # agent decides to keep going against wall
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 1, 0, current_state[3] + 1], dtype='int32')): # randomly right
                        return 0.5
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 3, 0, current_state[3] - 1], dtype='int32')): # randomly left
                        return 0.5
                    else:
                        return 0
                elif action == 1: # even if not trying to force into wall, we must take care not to let wind do it in the "standard case"
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 1, 0, current_state[3] + 1], dtype='int32')):
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 3, 0, current_state[3] - 1], dtype='int32')):
                        return 1 - self.direction_probability
                    else:
                        return 0
                else:
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 1, 0, current_state[3] + 1], dtype='int32')):
                        return 1 - self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 3, 0, current_state[3] - 1], dtype='int32')):
                        return self.direction_probability
                    else:
                        return 0
            elif current_state[1] == 1: # heading right
                if action == 0 or action == 2: # just proceeding OR trying to force into wall: same result.
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 1, 0, current_state[3] + 1], dtype='int32')): # case where it proceeds
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 0, 1, current_state[3]], dtype='int32')): # case where it gets blown down
                        return 1 - self.direction_probability
                    else:
                        return 0
                else: # trying to turn away from the wall; return opposite probabilities
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 1, 0, current_state[3] + 1], dtype='int32')): # case where it proceeds
                        return 1 - self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 0, 1, current_state[3]], dtype='int32')): # case where it gets blown down
                        return self.direction_probability
                    else:
                        return 0
            elif current_state[1] == 3: # heading left
                if action == 0 or action == 1: # just proceeding OR trying to force into wall: same result.
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 3, 0, current_state[3] - 1], dtype='int32')): # case where it proceeds
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 0, 1, current_state[3]], dtype='int32')): # case where it gets blown down
                        return 1 - self.direction_probability
                    else:
                        return 0
                else: # trying to turn away from the wall; return opposite probabilities
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 3, 0, current_state[3] - 1], dtype='int32')): # case where it proceeds
                        return 1 - self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 0, 1, current_state[3]], dtype='int32')): # case where it gets blown down
                        return self.direction_probability
                    else:
                        return 0
            else: # irrelevant impossible state
                if np.array_equal(successor_state, self.terminal_state):
                    return 1
                else:
                    return 0
        if current_state[2] == self.grid_size - 1: # bottom boundary
            if current_state[1] == 0: # heading down, must be forced to turn
                if action == 0: # agent decides to keep going against wall
                    if   np.array_equal(successor_state, np.array([current_state[0] - 1, 1, self.grid_size - 1, current_state[3] + 1], dtype='int32')): # randomly global right
                        return 0.5
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 3, self.grid_size - 1, current_state[3] - 1], dtype='int32')): # randomly global left
                        return 0.5
                    else:
                        return 0
                elif action == 1: # even if not trying to force into wall, we must take care not to let wind do it in the "standard case"
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 3, self.grid_size - 1, current_state[3] - 1], dtype='int32')):
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 1, self.grid_size - 1, current_state[3] + 1], dtype='int32')):
                        return 1 - self.direction_probability
                    else:
                        return 0
                else:
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 3, self.grid_size - 1, current_state[3] - 1], dtype='int32')):
                        return 1 - self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 1, self.grid_size - 1, current_state[3] + 1], dtype='int32')):
                        return self.direction_probability
                    else:
                        return 0
            elif current_state[1] == 1: # heading right
                if action == 0 or action == 1: # just proceeding OR trying to force into wall: same result.
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 1, self.grid_size - 1, current_state[3] + 1], dtype='int32')): # case where it proceeds
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 2, self.grid_size - 2, current_state[3]], dtype='int32')): # case where it gets blown up
                        return 1 - self.direction_probability
                    else:
                        return 0
                else: # trying to turn away from the wall; return opposite probabilities
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 1, self.grid_size - 1, current_state[3] + 1], dtype='int32')): # case where it proceeds
                        return 1 - self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 2, self.grid_size - 2, current_state[3]], dtype='int32')): # case where it gets blown up
                        return self.direction_probability
                    else:
                        return 0
            elif current_state[1] == 3: # heading left
                if action == 0 or action == 2: # just proceeding OR trying to force into wall: same result.
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 3, self.grid_size - 1, current_state[3] - 1], dtype='int32')): # case where it proceeds
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 2, self.grid_size - 2, current_state[3]], dtype='int32')): # case where it gets blown up
                        return 1 - self.direction_probability
                    else:
                        return 0
                else: # trying to turn away from the wall; return opposite probabilities
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 3, self.grid_size - 1, current_state[3] - 1], dtype='int32')): # case where it proceeds
                        return 1 - self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 2, self.grid_size - 2, current_state[3]], dtype='int32')): # case where it gets blown up
                        return self.direction_probability
                    else:
                        return 0
            else:
                if np.array_equal(successor_state, self.terminal_state):
                    return 1
                else:
                    return 0
        if current_state[3] == 0: # left boundary
            if current_state[1] == 3: # heading left, must be forced to turn
                if action == 0: # agent tries to force straight into wall
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 2, current_state[2] - 1, 0], dtype='int32')): # randomly up
                        return 0.5
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 0, current_state[2] + 1, 0], dtype='int32')): # randomly down
                        return 0.5
                    else:
                        return 0
                elif action == 1: # even if not trying to force into wall, we must take care not to let wind do it in the "standard case"
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 2, current_state[2] - 1, 0], dtype='int32')):
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 0, current_state[2] + 1, 0], dtype='int32')):
                        return 1 - self.direction_probability
                    else:
                        return 0
                else:
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 2, current_state[2] - 1, 0], dtype='int32')):
                        return 1 - self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 0, current_state[2] + 1, 0], dtype='int32')):
                        return self.direction_probability
                    else:
                        return 0
            elif current_state[1] == 1: # heading up
                if action == 0 or action == 2: # just proceeding OR trying to force into wall: same result.
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 2, current_state[2] - 1, current_state[3]], dtype='int32')): # case where it proceeds
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 1, current_state[2], 1], dtype='int32')): # case where it gets blown right
                        return 1 - self.direction_probability
                    else:
                        return 0
                else: # trying to turn away from the wall; return opposite probabilities
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 2, current_state[2] - 1, current_state[3]], dtype='int32')): # case where it proceeds
                        return 1 - self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 1, current_state[2], 1], dtype='int32')): # case where it gets blown right
                        return self.direction_probability
                    else:
                        return 0
            elif current_state[1] == 0: # heading down
                if action == 0 or action == 1: # just proceeding OR trying to force into wall: same result.
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 0, current_state[2] + 1, 0], dtype='int32')): # case where it proceeds
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 1, current_state[2], 1], dtype='int32')): # case where it gets blown right
                        return 1 - self.direction_probability
                    else:
                        return 0
                else: # trying to turn away from the wall; return opposite probabilities
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 0, current_state[2] + 1, 0], dtype='int32')): # case where it proceeds
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 1, current_state[2], 1], dtype='int32')): # case where it gets blown right
                        return 1 - self.direction_probability
                    else:
                        return 0
            else: # irrelevant impossible state
                if np.array_equal(successor_state, self.terminal_state):
                    return 1
                else:
                    return 0
        if current_state[3] == self.grid_size - 1: # right boundary
            if current_state[1] == 1: # heading right, must be forced to turn
                if action == 0:
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 2, current_state[2] - 1, self.grid_size - 1], dtype='int32')): # randomly up
                        return 0.5
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 0, current_state[2] + 1, self.grid_size - 1], dtype='int32')): # randomly down
                        return 0.5
                    else: 
                        return 0
                elif action == 1: # even if not trying to force into wall, we must take care not to let wind do it in the "standard case"
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 0, current_state[2] + 1, self.grid_size - 1], dtype='int32')):
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 2, current_state[2] - 1, self.grid_size - 1], dtype='int32')):
                        return 1 - self.direction_probability
                    else:
                        return 0
                else:
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 0, current_state[2] + 1, self.grid_size - 1], dtype='int32')):
                        return 1 - self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 2, current_state[2] - 1, self.grid_size - 1], dtype='int32')):
                        return self.direction_probability
                    else:
                        return 0
            elif current_state[1] == 1: # heading up
                if action == 0 or action == 1: # just proceeding OR trying to force into wall: same result.
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 2, current_state[2] - 1, current_state[3]], dtype='int32')): # case where it proceeds
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 3, current_state[2], current_state[3] - 1], dtype='int32')): # case where it gets blown left
                        return 1 - self.direction_probability
                    else:
                        return 0
                else: # trying to turn away from the wall; return opposite probabilities
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 2, current_state[2] - 1, current_state[3]], dtype='int32')): # case where it proceeds
                        return 1 - self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 3, current_state[2], current_state[3] - 1], dtype='int32')): # case where it gets blown left
                        return self.direction_probability
                    else:
                        return 0
            elif current_state[1] == 0: # heading down
                if action == 0 or action == 2: # just proceeding OR trying to force into wall: same result.
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 0, current_state[2] + 1, current_state[3]], dtype='int32')): # case where it proceeds
                        return self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 3, current_state[2], current_state[3] - 1], dtype='int32')): # case where it gets blown left
                        return 1 - self.direction_probability
                    else:
                        return 0
                else: # trying to turn away from the wall; return opposite probabilities
                    if np.array_equal(successor_state, np.array([current_state[0] - 1, 0, current_state[2] + 1, current_state[3]], dtype='int32')): # case where it proceeds
                        return 1 - self.direction_probability
                    elif np.array_equal(successor_state, np.array([current_state[0] - 1, 3, current_state[2], current_state[3] - 1], dtype='int32')): # case where it gets blown left
                        return self.direction_probability
                    else:
                        return 0
            else: # impossible useless situation
                if np.array_equal(successor_state, self.terminal_state):
                    return 1
                else:
                    return 0

        # Now, consider the most common case - normal descending flight.
        # The stochastics array describes the probability, given an action from (0,1,2), of the result corresponding to what we'd expect from each of those actions
        # if action == 1, for example, if stochastics == array[0.05,0.9,0.05], then the resulting successor state will be what we would expect of action == 1 with 90% probability,
        # and 5% probability for each of the other directions.
        # notice the random directions are contained in forward,right,left WITH RESPECT TO the agent's current heading. So if the agent tries to turn left, it can
        # actually end up going left, forward or right w.r.t. its state just before the action.  
        stochastics = np.full(len(self.action_space), self.prob_other_directions)
        stochastics[action] = self.direction_probability
        # if the successor_state is reachable from current_state, we return the probabilities of getting there, given our input action
        # these probabilities have been defined by the stochastics vector above
        
        successor_probability = 0 # initialise probability of successor, which might in the end be sum of various components of stochastics vector due to environment boundaries.
        for direction_number in range(len(self.action_space)):
            direction = self.action_to_direction[current_state[1]][direction_number] # iterate over the 2-element direction vectors
            
            potential_successor = np.zeros(4, dtype='int32') # initialise
            potential_successor[2:] = np.clip(current_state[2:] + direction, 0, self.grid_size - 1) # assign 2D component, as in 2D version
            potential_successor[0] = current_state[0] - 1 # assign altitude as current's - 1
            potential_successor[1] = self.direction_to_heading(direction) # new heading is just equal to the direction the aircraft has travelled in.

            # if the direction would lead us from current_state to successor_state, add to the output the probability
            # that the action given would lead us to that direction.
            if np.array_equal(potential_successor, successor_state):
                successor_probability += stochastics[direction_number] 
        return successor_probability
        

    # this is where the dynamics are SAMPLED.
    # returns a sample of the successor state given a current state and an action, as well as the reward from the successor
    # it's NOT used in the dynamic programming algorithms because those require the actual probability distributions of state transitions as functions of actions.
    # will be used if we move onto Monte Carlo methods or to just run individual episodes of the environment/agent/policy.
    def state_transition(self, state, action):
        if state[0] <= 0: # crashed, terminal, or landed
            new_state = self.terminal_state
            return new_state
        
        # consider obstacle cases, similar to crashed cases.
        for obstacle in self.obstacles:
            if np.array_equal(state[2:], obstacle):
                new_state = self.terminal_state
                return new_state
        
        # we're left with the case in flight, with actions being one of (0,1,2,3,4)
        # the stochastics array describes the probability, given an action from (0,1,2,3,4), of the result corresponding to what we'd expect from each of those actions
        # if action == 1, for example, if stochastics == array[0.05,0.8,0.05,0.05,0.05], then the resulting successor state will be what we would expect of action == 1 with 80% probability,
        # and 5% probability for each of the other directions.
        

        # first consider the CORNERS.
        if (state[2] == 0 and state[3] == 0): # top-left corner
            if state[1] == 2: # heading up
                new_state = np.array([state[0] - 1, 1, 0, 1], dtype='int32') # turn to global right
                return new_state
            elif state[1] == 3: # heading left
                new_state = np.array([state[0] - 1, 0, 1, 0], dtype='int32') # turn to global down
                return new_state
        if (state[2] == 0 and state[3] == (self.grid_size - 1)): # top-right corner
            if state[1] == 1: # heading right
                new_state = np.array([state[0] - 1, 0, 1, self.grid_size - 1], dtype='int32') # turn to global down
                return new_state
            elif state[1] == 2: # heading up
                new_state = np.array([state[0] - 1, 3, 0, self.grid_size - 2], dtype='int32') # turn to global left
                return new_state
        if (state[2] == (self.grid_size - 1) and state[3] == 0): # bottom-left corner
            if state[1] == 0: # heading down
                new_state = np.array([state[0] - 1, 1, self.grid_size - 1, 1], dtype='int32') # turn to global right
                return new_state
            elif state[1] == 3: # heading left
                new_state = np.array([state[0] - 1, 2, self.grid_size - 2, 0], dtype='int32') # turn to global up
                return new_state
        if (state[2] == (self.grid_size - 1) and state[3] == (self.grid_size - 1)): # bottom-right corner
            if state[1] == 0: # heading down
                new_state = np.array([state[0] - 1, 3, self.grid_size - 1, self.grid_size - 2], dtype='int32') # turn to global left
                return new_state
            elif state[1] == 1: # heading right
                new_state = np.array([state[0] - 1, 2, self.grid_size - 2, self.grid_size - 1], dtype='int32') # turn to global up
                return new_state
        

        stochastics = np.full(len(self.action_space), self.prob_other_directions)
        stochastics[action] = self.direction_probability
        # now consider NON-CORNER BOUNDARIES
        if state[2] == 0: # top boundary
            if state[1] == 2: # heading up, must be forced to turn
                if action == 0: # agent decides to keep going against wall
                    stochastics = np.array([0, 0.50, 0.50])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
                elif action == 1: # even if not trying to force into wall, we must take care not to let wind do it in the "standard case"
                    stochastics = np.array([0, self.direction_probability, 1 - self.direction_probability])
                else:
                    stochastics = np.array([0, 1 - self.direction_probability, self.direction_probability])
            elif state[1] == 1: # heading right
                if action == 0 or action == 2: # just proceeding OR trying to force into wall: same result.
                    stochastics = np.array([self.direction_probability, 1 - self.direction_probability, 0])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
                else: # trying to turn away from the wall; return opposite probabilities
                    stochastics = np.array([1 - self.direction_probability, self.direction_probability, 0])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
            elif state[1] == 3: # heading left
                if action == 0 or action == 1: # just proceeding OR trying to force into wall: same result.
                    stochastics = np.array([self.direction_probability, 0, 1 - self.direction_probability])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
                else: # trying to turn away from the wall; return opposite probabilities
                    stochastics = np.array([1 - self.direction_probability, 0, self.direction_probability])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
        if state[2] == self.grid_size - 1: # bottom boundary
            if state[1] == 0: # heading down, must be forced to turn
                if action == 0: # agent decides to keep going against wall
                    stochastics = np.array([0, 0.50, 0.50])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
                elif action == 1: # even if not trying to force into wall, we must take care not to let wind do it in the "standard case"
                    stochastics = np.array([0, self.direction_probability, 1 - self.direction_probability])
                else:
                    stochastics = np.array([0, 1 - self.direction_probability, self.direction_probability])
            elif state[1] == 1: # heading right
                if action == 0 or action == 1: # just proceeding OR trying to force into wall: same result.
                    stochastics = np.array([self.direction_probability, 0, 1 - self.direction_probability])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
                else: # trying to turn away from the wall; return opposite probabilities
                    stochastics = np.array([1 - self.direction_probability, 0, self.direction_probability])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
            elif state[1] == 3: # heading left
                if action == 0 or action == 2: # just proceeding OR trying to force into wall: same result.
                    stochastics = np.array([self.direction_probability, 1 - self.direction_probability, 0])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
                else: # trying to turn away from the wall; return opposite probabilities
                    stochastics = np.array([1 - self.direction_probability, self.direction_probability, 0])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
        if state[3] == 0: # left boundary
            if state[1] == 3: # heading left, must be forced to turn
                if action == 0: # agent decides to keep going against wall
                    stochastics = np.array([0, 0.50, 0.50])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
                elif action == 1: # even if not trying to force into wall, we must take care not to let wind do it in the "standard case"
                    stochastics = np.array([0, self.direction_probability, 1 - self.direction_probability])
                else:
                    stochastics = np.array([0, 1 - self.direction_probability, self.direction_probability])
            elif state[1] == 0: # heading down
                if action == 0 or action == 1: # just proceeding OR trying to force into wall: same result.
                    stochastics = np.array([self.direction_probability, 0, 1 - self.direction_probability])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
                else: # trying to turn away from the wall; return opposite probabilities
                    stochastics = np.array([1 - self.direction_probability, 0, self.direction_probability])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
            elif state[1] == 2: # heading up
                if action == 0 or action == 2: # just proceeding OR trying to force into wall: same result.
                    stochastics = np.array([self.direction_probability, 1 - self.direction_probability, 0])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
                else: # trying to turn away from the wall; return opposite probabilities
                    stochastics = np.array([1 - self.direction_probability, self.direction_probability, 0])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
        if state[3] == self.grid_size - 1: # right boundary
            if state[1] == 1: # heading right, must be forced to turn
                if action == 0: # agent decides to keep going against wall
                    stochastics = np.array([0, 0.50, 0.50])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
                elif action == 1: # even if not trying to force into wall, we must take care not to let wind do it in the "standard case"
                    stochastics = np.array([0, self.direction_probability, 1 - self.direction_probability])
                else:
                    stochastics = np.array([0, 1 - self.direction_probability, self.direction_probability])
            elif state[1] == 0: # heading down
                if action == 0 or action == 2: # just proceeding OR trying to force into wall: same result.
                    stochastics = np.array([self.direction_probability, 1 - self.direction_probability, 0])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
                else: # trying to turn away from the wall; return opposite probabilities
                    stochastics = np.array([1 - self.direction_probability, self.direction_probability, 0])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
            elif state[1] == 2: # heading up
                if action == 0 or action == 1: # just proceeding OR trying to force into wall: same result.
                    stochastics = np.array([self.direction_probability, 0, 1 - self.direction_probability])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state
                else: # trying to turn away from the wall; return opposite probabilities
                    stochastics = np.array([1 - self.direction_probability, 0, self.direction_probability])
                    effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
                    effective_direction = self.action_to_direction[state[1]][effective_action]
                    new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
                    new_heading = self.direction_to_heading(effective_direction)
                    new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
                    return new_state

        # finally we consider a normal case not on the boundaries of the problem
        effective_action = self.rng.choice(len(self.action_space), p=stochastics) # this is the effective action, after sampling from the action-biased distribution. Most times this should be equal to intended action
        effective_direction = self.action_to_direction[state[1]][effective_action]
        new_state_2d = np.clip(state[2:] + effective_direction, 0, self.grid_size - 1) # gives new state, just in the horizontal plane, missing altitude
        new_heading = self.direction_to_heading(effective_direction)
        new_state = np.concatenate((np.array([state[0] - 1], ndmin=1), np.array(new_heading, ndmin=1), new_state_2d))
        
        return new_state

# epsilon = the threshold delta must go below in order for us to stop
# value function is held in a column vector of size equal to len(MDP.state_space)
def policy_evaluation(policy, MDP, initial_value, epsilon=0, max_iterations=50):
    if initial_value is None:
        current_value = np.zeros(MDP.problem_shape) # default initial guess is all zeros
    else:
        current_value = initial_value

    change = np.zeros(MDP.problem_shape) # this will store the change in the value for each state, in the latest iteration
    # delta will always be positive after starting iterations (it's an absolue value).
    # thus we initialise it to -1 so that it doesn't trigger the while condition right away.
    delta = -1 # initialising the variable that will store the max change in the value_function across all states
    iteration_no = 1
    while (delta < 0 or delta > epsilon) and iteration_no <= max_iterations:
        """
        print(f'Iteration number: {iteration_no}')
        print()
        print('Current value function estimate:')
        print(current_value[:MDP.max_altitude + 1])
        print()
        """

        # in 2D, we indexed the value function data structure by the raw state (then a 2D vector).
        # In 3D we have to switch to indexing by a single number, because the value is stored in a column vector.
        for state in MDP.state_space:
            old_state_value = current_value[tuple(state)]
            current_value_update = 0
            for action in MDP.action_space:
                sub_sum = 0

                possible_successors = accessible_states(state, MDP)
                for successor in possible_successors:
                    # CRUCIAL NOTE
                    # in the below line, I changed (as of 25/01/2023) what was MDP.reward(successor) to MDP.reward(state)
                    # this made the algorithms work towards optimal policies for the problem as of 25/01/2023, but change back if needed.
                    # SEE for-meeting14.md in UoB repo FOR DETAILS
                    sub_sum += MDP.environment_dynamics(successor, state, action) * (MDP.reward(state) + MDP.discount_factor * current_value[tuple(successor)])




                current_value_update += policy(action,state) * sub_sum
            current_value[tuple(state)] = current_value_update
            change[tuple(state)] = abs(current_value[tuple(state)] - old_state_value)
        delta = change.max()

        """
        print('Absolute changes to value function estimate:')
        print(change[:MDP.max_altitude + 1])
        print()
        print()
        print()
        print()
        """

        iteration_no += 1
    return current_value

# the below returns whether successor_state is in principle reachable from current_state, given the gridworld assumption
# of a single rectangular move per time step in the grid environment domain
# THIS IS NOT ADAPTED TO 3D, BECAUSE WE DON'T USE IT IN THE ALGORITHMS AS OF 06/02/2023
def is_accessible(current_state, successor_state):
    state_difference = successor_state - current_state
    if np.array_equal(state_difference, [1,0]) or np.array_equal(state_difference, [-1,0]) or np.array_equal(state_difference, [0,1]) or np.array_equal(state_difference, [0,-1]):
        return True
    else:
        return False

# this function returns all states that are accessible from the current_state of the agent
# since this is intended for generating greedy policies and other useful stuff, we'll rule out the agent's own state, even when that is accessible (e.g., by trying to move outside of the domain boundaries).
# as it stands, this function is expensive because it keeps stacking rows -- appending, which requires copying array each time!
def accessible_states(current_state, MDP):
    
    # first we consider the special cases:
    # agent has landed or
    # agent has crashed.
    # in these cases, the only accessible state from there is the MDP's terminal state.
    if current_state[0] <= 0:
        return np.array(MDP.terminal_state, ndmin=2)
    
    for obstacle in MDP.obstacles:
        if np.array_equal(current_state[2:], obstacle):
            return np.array(current_state, ndmin=2)

    # now onto all other cases, where aircraft is still flying.
    # initialise output to be zeros of shape (len(MDP.action_space), 4).
    # 3 columns because 3D state vector.
    # NUMBER OF ROWS determined by the number of actions available to the agent.
    # for instance, with 3 actions (forward, right, left), there are, from any given state, AT MOST 3 different states the agent can come to occupy.
    # thus, the output of this function might contain duplicate states, e.g., if current_state is at a boundary of the grid.
    # but that's okay for the purposes of the function.
    output = np.zeros(shape=(len(MDP.action_space), 4), dtype='int32')
    
    # we are not accounting for the more restrictive boundary cases, but it's okay, the purpose of this function is to just cut down on the search space.
    for action in range(len(MDP.action_space)):
        direction = MDP.action_to_direction[current_state[1]][action] # 2-element direction vector
        potential_accessible = np.zeros(4, dtype='int32') # initialise

        potential_accessible[2:] = np.clip(current_state[2:] + direction, 0, MDP.grid_size - 1)
        potential_accessible[0] = current_state[0] - 1 # assign altitude as current's - 1
        potential_accessible[1] = MDP.direction_to_heading(direction) # new heading is just equal to the direction the aircraft has travelled in.

        output[action] = potential_accessible
    
    return output

# this returns a 2D array with integers codifying greedy actions in it, with respect to an input value function.
# from this, still need to construct a policy as a function policy(action, state), which returns a probability distribution over actions, given some current state.
# keep in mind that such a greedy policy will always be deterministic, so the probability distribution will be very boring, with 1 assigned to the greedy action and 0 elsewhere.
# however, this general structure is useful as it can be used directly in the generalised policy evaluation algorithm we've implemented, which assumes that form of a policy(action, state).
def greedy_policy_array(value_function, MDP):
    policy_array = np.empty(shape=MDP.problem_shape, dtype='int32') # this array stores actions (0,1,2,3) which codify the greedy policy
    for state in MDP.state_space:
        potential_next_states = accessible_states(state, MDP)
        max_next_value = np.NINF # initialise max value attainable as minus infinity
        for successor_state in potential_next_states:
            potential_value = value_function[tuple(successor_state)]
            if potential_value > max_next_value:
                greedy_state_difference = successor_state - state
                max_next_value = potential_value
        policy_array[tuple(state)] = MDP.state_difference_to_action(greedy_state_difference, state)
    return policy_array

# take array of scalar action representations and transform it into an actual policy(action, state)
def array_to_policy(policy_array, MDP):
    # 5D array used
    # 2nd, 3rd, 4th and 5th indices correspond to dimensions of the state space.
    # 1st index corresponds to action number.
    state_action_probabilities = np.zeros(shape = (len(MDP.action_space), MDP.problem_shape[0], MDP.problem_shape[1], MDP.problem_shape[2], MDP.problem_shape[3]))
    for index in np.ndindex(MDP.problem_shape[0], MDP.problem_shape[1], MDP.problem_shape[2], MDP.problem_shape[3]):
        greedy_action = policy_array[index]
        
        state_action_probabilities[(greedy_action,) + index] = 1 # deterministic policy: just set probability of a single action to 1
    
    # policy function itself just has to index the 3D array we've created, which contains all the policy-defining information
    def policy(action, state):
        return state_action_probabilities[(action,) + tuple(state)]
    
    # return policy function
    return policy

# fuse greedy_policy_array and array_to_policy.
# go directly from value function to greedy policy as function of action and state
def value_to_greedy_policy(value_function, MDP):
    policy_array = greedy_policy_array(value_function, MDP)
    return array_to_policy(policy_array, MDP)

# carry out policy iteration up to some limit number of iterations, or until policy stabilises
# policy_evaluation(policy, MDP, epsilon, max_iterations)
def policy_iteration(policy, MDP, evaluation_max_iterations=50, improvement_max_iterations=50, train_time=False):
    iteration_count = 1
    policy_is_stable = False
    current_policy = policy
    initial_value = None
    current_policy_array = np.ones(shape=MDP.problem_shape, dtype='int32') * -10 # initialise greedy policy array to a bogus instance

    st = time.time()
    while policy_is_stable is False and iteration_count <= improvement_max_iterations:
        print(f'Iteration number: {iteration_count}')
        print(f'Terminal state: {MDP.terminal_state}')
        print('Current greedy policy array (disregard in iteration no. 1):')
        print(current_policy_array[:MDP.max_altitude + 1])
        print()

        # as per Sutton Barto 2nd, chapter 4.3, next iteration is better-converging if we
        # start with the previous value estimate, hence the assignment into initial_value.
        initial_value = policy_evaluation(current_policy, MDP, initial_value, epsilon=0, max_iterations=evaluation_max_iterations)
        print('Previous policy evaluation:')
        print(initial_value[:MDP.max_altitude + 1])
        new_policy_array = greedy_policy_array(initial_value, MDP)
        
        if np.array_equal(new_policy_array, current_policy_array):
            policy_is_stable = True
            print('Policy has stabilised.')
            print()
            break # stop iterating

        current_policy_array = new_policy_array
        current_policy = array_to_policy(new_policy_array, MDP)
        iteration_count += 1
    
    et = time.time()
    print('Final policy array:')
    print(current_policy_array[:MDP.max_altitude + 1])
    print()
    if train_time is False:
        return current_policy, current_policy_array
    else:
        training_time = et - st
        return current_policy, current_policy_array, training_time

def value_iteration(policy, MDP, max_iterations, train_time=False):
    return policy_iteration(policy, MDP, evaluation_max_iterations=1, improvement_max_iterations=max_iterations, train_time=train_time)

# input policy to evaluate
def run_policy_evaluation(use_policy):
    os.system('clear')
    default = input('Run policy evaluation with default parameters? (y/n) ')
    if default.split()[0][0].upper() == 'Y':
        grid_size = 3
        direction_probability = 1
        max_altitude = 4
        max_iterations = 15
        epsilon = 0
    else:
        grid_size = int(input('Input grid size: '))
        direction_probability = float(input('Input probability of action success: '))
        max_iterations = int(input('Input max number of iterations: '))
        epsilon = float(input('Input epsilon for convergence: '))

    GridWorld = MarkovGridWorld(grid_size=grid_size, direction_probability=direction_probability, max_altitude=max_altitude)
    print('-----------------------------------------------------------------------------')
    print('Running policy evaluation.')
    print(f'Grid size: {GridWorld.grid_size}')
    print(f'Max altitude: {GridWorld.max_altitude}')
    print(f'Terminal state: {GridWorld.terminal_state}')
    print(f'Landing zone: {GridWorld.landing_zone}')
    print(f'Obstacles: {GridWorld.obstacles}')
    print(f'Discount factor: {GridWorld.discount_factor}')
    print(f'Probability of action resulting in intended direction of motion: {GridWorld.direction_probability}')
    print(f'Epsilon: {epsilon}')
    print(f'Max iterations: {max_iterations}')
    print(f'Policy: {use_policy}')
    print('-----------------------------------------------------------------------------')
    input('Press Enter to continue...')
    print()
    value = policy_evaluation(policy = use_policy, MDP = GridWorld, initial_value = None, epsilon = epsilon, max_iterations=max_iterations)
    #greedy_policy_scalars = greedy_policy_array(value, GridWorld)
    #greedy_policy = array_to_policy(greedy_policy_scalars, GridWorld)
    print('-----------------------------------------------------------------------------')
    print()
    print()
    print()
    print()
    print('Final value estimation:')
    print(value[1:])
    print()
    #print('Greedy policy array representation with respect to final value function estimate:')
    #print(greedy_policy_scalars[:GridWorld.max_altitude + 1])

def run_value_iteration(policy, MDP, print_bool, max_iterations=1000):
    os.system('clear')

    if print_bool:
        print('-----------------------------------------------------------------------------')
        print('Running value iteration.')
        print(f'Grid size: {MDP.grid_size}')
        print(f'Max altitude: {MDP.max_altitude}')
        print(f'Terminal state: {MDP.terminal_state}')
        print(f'Discount factor: {MDP.discount_factor}')
        print(f'Probability of action resulting in intended direction of motion: {MDP.direction_probability}')
        print(f'Max iterations: {max_iterations}')
        print(f'Policy: {policy}')
        print('-----------------------------------------------------------------------------')
        input('Press Enter to continue...')
        print()


    st = time.time()
    optimal_policy, optimal_policy_array = value_iteration(policy, MDP, max_iterations=max_iterations)
    et = time.time()
    elapsed_time = et - st
    if print_bool:
        print(f'Elapsed time: {elapsed_time} seconds')

def run_policy_iteration(policy, MDP, max_evaluation_iterations, max_improvement_iterations):
    policy_iteration(policy, MDP, max_evaluation_iterations, max_improvement_iterations)


def run_profiler(function):
    import cProfile
    cProfile.run(function, 'output.dat') # function is given as a string (e.g., 'policy_evaluation()')

    import pstats
    from pstats import SortKey

    with open("output_time.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("time").print_stats()

    with open("output_calls.txt", "w") as f:
        p = pstats.Stats("output.dat", stream=f)
        p.sort_stats("calls").print_stats()

if __name__ == "__main__":
    import monte_carlo_4d as mc4
    with open("benchmark-problems/4d/42_wind_0,9.p", 'rb') as f:
            train_MDP = pickle.load(f) # load MDP for training (with correct wind parameter as decided to be used during training)

    optimal_policy, optimal_policy_array, training_time = value_iteration(random_walk, train_MDP, np.inf, train_time=True)
    print(training_time)
    mc4.simulate_policy(train_MDP, optimal_policy, 10)
    
    
