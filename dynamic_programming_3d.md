# Functions

## ```test_policy(action, state)```

Defines a policy to be used as a starting point. Returns a probability of selecting action given a state. Defined as just random action (returns 0.20 no matter the state, no matter the action).

See also some alternative simple policies defined for testing purposes.

## ```policy_evaluation(policy, MDP, initial_value, epsilon, max_iterations)```

Returns estimate of value as a function of state, following a given policy. Inputs are the policy, an MDP (Markov Decision Process) definition, and parameters for iteration termination (epsilon is used as a error tolerance at which point no more iterations are done; otherwise, stops when max_iterations have been performed).

```initial_value``` is the initial guess of the value function in the entire domain. Having this as an explicit argument (as opposed to, say, initialising it to a zeros array every time) is useful because, in policy (and value) iteration, using ```initial_value``` equal to our estimate of the value function given the previous policy typically results in a great increase in the speed of convergence of policy evaluation (see Sutton and Barto 2nd, Chapter 4.3).

As of 19/01/2023, the entire function relies on explicit ```for``` loops to run the algorithm. A potential goal would be to vectorise all of these operations, thus getting a big performance boots, but this requires a lot of hard thinking about complicated multi-dimensional arrays. Thus this is put behind the bigger priority of building reasonable algorithms that work reasonably well on models reasonably resembling the real problem scenario of interest.

## ```is_accessible(current_state, successor_state)```

**NOTE**: As of 23/01/2023, not adapted to 3D, but not used anywhere else either.

Given a starting and potential final state of interest, returns whether ```successor_state``` is 'within reach' of ```current_state```. "Within reach" encompassess all states within 1 grid cell of ```current_state```, and no more — thus if ```current_state == successor_state```, the function returns ```False```.

As of 19/01/2023, it's important to note that the function is designed in a way that works exclusively for an MDP which consists of a typical 2D grid world. The function returns values based on explicit possible values for the difference between 2D initial and final states.

## ```accessible_states(current_state, MDP)```

Returns an m x n matrix, where m is the number of accessible states and n is the dimension of the MDP grid world.

## ```greedy_policy_array(value_function, MDP)```

Returns an array with the size equal to the MDP's grid world environment, where entries give the greedy policy with respect to the input value function. That is, each scalar entry corresponds to an action which the agent should take aiming to end up at the state accessible to it with the highest value.

Crucial for policy and value iteration.

## ```array_to_policy(policy_array, MDP)```

Takes array of (as of 19/01/2023, deterministic) greedy actions as output by ```greedy_policy_array``` and returns an actual function ```policy(action, state)``` in its most general format, to be used as before in the developed algorithms.

## ```value_to_greedy_policy()```

Returns greedy policy (function of action, state) given a value function. Basically just a chaining together of ```greedy_policy_array```and ```array_to_policy```.

## ```policy_iteration(policy, MDP, evaluation_max_iterations=10, improvement_max_iterations=10)```

Returns array representation of policy that comes out of policy iteration algorithm.

The policy iteration algorithm first evaluates the current policy via value evaluation (itself an iterative algorithm which runs up to a maximum of ```evaluation_max_iterations```), defines a new and improved greedy policy, and repeates the whole process up to ```improvement_max_iterations``` times. Outer (improvement) loop breaks if the new greedy policy array representation is equal to the current one, meaning the policy has stabilised and is, therefore, optimal (see Sutton, Barto 2nd, 4.3).

## ```value_iteration(policy, MDP, max_iterations)```

Returns array representation of policy that comes out of value iteration algorithm.

The value iteration algorithm is the same as the policy iteration algorithm in the case where we truncate value evaluation to a single iteration. Thus, the function is equivalent to ```policy_iteration(policy, MDP, evaluation_max_iterations=1, improvement_max_iterations=max_iterations)```.

As of 20/01/2023, from a uniformly random initial policy, policy stabilisation on 10x10 grid world takes approximately 21 seconds. For 20x20 grid world, takes approximately 362 seconds / 6 minutes.

Around this size range, given an NxN grid world environment, the algorithm execution time appears to scale as something between O(N<sup>3</sup>) and O(N<sup>4</sup>), depending on N.

## ```run_policy_evaluation()```

Actually runs algorithm of interest (in this case, policy evaluation) using the functions and classes defined before,  while displaying useful information, as well as other things such as writing profiler output.

# Classes

## ```MarkovGridWorld()```

Defines the 3D grid world MDP. This includes defining the environment's size, state space (and terminal state), action space, dynamics (including any potential stochastics), reward signals, and discount factor.

### Attributes

#### ```MDP.state_space```

The state space is a matrix composed of stacked 3-element np.arrays, each representing a state. Each of these arrays is of the form [altitude, x, y]. Whatever the initial state, all actions (except for the landing action) lead the agent's altitude to decrease by 1. Otherwise the actions move the agent about in the xy plane as before, with all the potential stochastics also implemented as before, etc.

#### ```MDP.action_space```

The 3D action space is a 5-element tuple (0,1,2,3,4,5). As before, in each horizontal plane we have 4 actions:

- 0 —> **stay put**.
- 1 —> down.
- 2 —> right.
- 3 —> up.
- 4 —> left.
- 5 —> **land**.

The 3D environment brings a new action numbered 4 (as well as 0 for staying put, but that could've been done in 2D as well). This is the ***landing*** action. The agent can take this action from any state with 0 < altitude < ```MDP.max_altitude```. Taking the landing action from state [altitude, x, y] leads the agent to state [altitude + ```MDP.max_altitude```, x, y]. The existence of these states with altitude above ```MDP.max_altitude``` serves the purpose of maintaining the reward signal as a function of state alone. In order to keep that structure we require a state that informs the algorithms that the agent has just landed at some point **from a known altitude**. Then, the reward signal is awarded based on the altitude from which the agent landed (landing from altitude 1 is the ideal case, the higher up the worse the reward should be), and in future the proximity to the prescribed landing zone might be taken into account too (as of 25/01/2023, only landing in the exact prescribed spot yields any reward).

Before 23/01/2023, the 'landing performed' state signal came from having negative altitudes. This, however, disallowed us from directly indexing the value function using the state (negative indices would make it all a mess). Thus the change to positive altitudes above the MDP-set ceiling, which still contains all the necessary information — easy to discern, and uniquely identifying the altitude when the landing manoeuvre was performed (via the modulo operator).

### Methods

#### ```MDP.direction_to_action(self,direction)```

Returns a 2-element array corresponding to an action in the horizontal plane given in terms of a scalar representation in 2D grid world.

#### ```MDP.reward(self, state)```

Returns reward signal as a function of just the state. As of 23/01/2023, reward of any state **except for** a landed state (altitude > ```MDP.max_altitude```) at prescribed landing zone (```MDP.landing_zone```) is 0. If the agent has landed at the location of a prescribed landing zone, the reward is higher the closer the agent was to the ground upon performing the landing manoeuvre. Thus the maximum reward is given to the state ```[MDP.max_altitude + 1, MDP.landing_zone]```.

#### ```MDP.environment_dynamics(self, successor_state, current_state, action)```

Returns the probability of the agent going from ```current_state``` to ```successor_state```, given ```action```.

This is one of the most important functions for carrying out dynamic programming algorithms (policy evaluation at the core of it) because it is kept in its most general and "omniscient" form: returning probabilities for successor states from current states given an action.

#### ```MDP.state_transition(self, state, action)```

***Samples*** the dynamics of the MDP, returning a **sampled successor state** as a function of the current state and an action. Thus, this will be useful for running simulations or for Monte Carlo methods, but ***not*** for dynamic programming methods.