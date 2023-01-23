# This piece of documentation, as of 23/01/2023, just mirrors that of the 2D dynamic programming documentation. This is not quite right for most functions/methods, so needs adaptation over to the 3D case, which is still in early development. Only some of the MDP methods' documentations have been crucially adapted.

# Functions

## ```test_policy(action, state)```

Defines a policy to be used as a starting point. Returns a probability of selecting action given a state. As of 19/01/2023, it's defined simply as returning equal probability of 0.25 for any of 4 actions in 2D (left, right, up down). As I progress from just policy evaluation to policy iteration, this will become basically irrelevant and just an "initial guess" to iterate on.

## ```policy_evaluation(policy, MDP, epsilon, max_iterations)```

**NOTE**: As of 23/01/2023, still needs adaptation for 3D. This will require changing the format of the value function, since it cannot anymore be represented by a 2D array that's easily visible. Will need to generalise it to just be a vector with length equal to the size of the state_space, and mostly forget about being able to print it out in an intuitive manner (could be possible, if challenging, with a 3D state, but if we are to add heading or any other additional dimension to the state, it becomes nigh-on impossible!).

Returns estimate of value function as a function of state, following a given policy. Inputs are the policy, an MDP (Markov Decision Process) definition, and parameters for iteration termination (epsilon is used as a error tolerance at which point no more iterations are done; otherwise, stops when max_iterations have been performed).

As of 19/01/2023, the entire function relies on explicit ```for``` loops to run the algorithm. A potential goal would be to vectorise all of these operations, thus getting a big performance boots, but this requires a lot of hard thinking about complicated multi-dimensional arrays. Thus this is put behind the bigger priority of building reasonable algorithms that work reasonably well on models reasonably resembling the real problem scenario of interest.

```current_value``` is the initial guess of the value function in the entire domain. this is an array of scalars (1 scalar per state in the gridworld) corresponding to our current estimate, which we iterate on: computing the value function is the ultimate purpose of policy evaluation.

## ```is_accessible(current_state, successor_state)```

Given a starting and potential final state of interest, returns whether ```successor_state``` is 'within reach' of ```current_state```. "Within reach" encompassess all states within 1 grid cell of ```current_state```, and no more — thus if ```current_state == successor_state```, the function returns ```False```.

As of 19/01/2023, it's important to note that the function is designed in a way that works exclusively for an MDP which consists of a typical 2D grid world. The function returns values based on explicit possible values for the difference between 2D initial and final states.

## ```accessible_states(current_state, MDP)```

Returns m x n matrix, where m is the number of accessible states and n is the dimension of the MDP grid world.

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

The state space is a matrix composed of stacked 3-element np.arrays, each representing a state. Each of these arrays is of the form [x,y,z]. Whatever the initial state, all actions (except for the landing action) lead the agent's z/altitude to decrease by 1. Otherwise the actions move the agent about in the xy plane as before, with all the potential stochastics also implemented as before, etc.

#### ```MDP.action_space```

The 3D action space is a 5-element tuple (0,1,2,3,4). As before, in each horizontal plane we have 4 actions:

- 0 —> down.
- 1 —> right.
- 2 —> up.
- 3 —> left.

The 3D environment brings a new action numbered 4. This is the ***landing*** action. The agent can take this action from any state with altitude > 0. Taking the landing action from state [x,y,altitude] leads the agent to state [x,y,-altitude]. The existence of these states with negative altitude serves the purpose of maintaining the reward signal as a function of state alone. In order to keep that structure we require a state that informs the algorithms that the agent has just landed at some point from a given altitude. Then, the reward signal is awarded based on the altitude from which the agent landed (landing from altitude 1 is the ideal case, the higher up the worse the reward should be), and in future the proximity to the prescribed landing zone might be taken into account too (as of 23/01/2023, only landing in the exact prescribed spot yields any reward).

### Methods

#### ```MDP.direction_to_action(self,direction)```

Returns a 2-element array corresponding to an action given in terms of a scalar representation in 2D grid world.

#### ```MDP.reward(self, state)```

Returns reward signal as a function of just the state. As of 23/01/2023, reward of any state except for a state landed (altitude < 0)at prescribed landing zone (```MDP.terminal_state```) is 0. If the agent has landed at the location of a prescribed landing zone, the reward is higher the closer the agent was to the ground upon performing the landing manoeuvre. Thus the maximum reward is given to the state ```[MDP.terminal_state, -1]```.

#### ```MDP.environment_dynamics(self, successor_state, current_state, action)```

Returns the probability of the agent going from ```current_state``` to ```successor_state```, given ```action```.
One of the most vital methods of the MarkovGridWorld() class for dynamic programming.

This is one of the most important functions for carrying out dynamic programming algorithms (policy evaluation at the core of it) because it is kept in its most general and omniscient form: returning probabilities for successor states from current states given an action.
This is in stark contrast to ```MDP.state_transition```, which is not in this general form. That method instead just samples the dynamics of the MDP, thus returning a sampled successor state as a function of just the current state and an action. Thus, this will be useful for running simulations or for Monte Carlo methods, but ***not*** for dynamic programming methods.

#### ```MDP.state_transition(self, state, action)```

Returns a successor state as a function of ```state``` and ```action```, by sampling of the environment's generally stochastic dynamics.

**NOTE**: This is (as of 23/01/2023) not yet adapted for 3D environment use. This is because it is not used in dynamic programming algorithms (```MDP.environment_dynamics``` is the method used for that), which is where the focus is for now. Will need to adapt this method if running individual episodes/simulations or resorting to Monte Carlo methods, which are based on averaging sampled agent experiences.