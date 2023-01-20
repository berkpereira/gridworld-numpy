# Functions

## ```test_policy(action, state)```

Defines a policy to be used as a starting point. Returns a probability of selecting action given a state. As of 19/01/2023, it's defined simply as returning equal probability of 0.25 for any of 4 actions in 2D (left, right, up down). As I progress from just policy evaluation to policy iteration, this will become basically irrelevant and just an "initial guess" to iterate on.

## ```policy_evaluation(policy, MDP, epsilon, max_iterations)```

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

As of 20/01/2023, from a uniformly random initial policy, policy stabilisation on 10x10 grid world takes approximately 21 seconds.

## ```run_policy_evaluation()```

Actually runs algorithm of interest (in this case, policy evaluation) using the functions and classes defined before,  while displaying useful information, as well as other things such as writing profiler output.

# Classes

## ```MarkovGridWorld()```

Defines the (as of 19/01/2023, 2D) grid world MDP. This includes defining the environment's size, state space (and terminal state), action space, dynamics (including any potential stochastics), reward signals, and discount factor.

0 is down.
1 is right.
2 is up.
3 is left.

### ```direction_to_action(self,direction)```

Returns a 2-element array corresponding to an action given in terms of a scalar representation in 2D grid world.

### ```reward(self, state)```

Returns reward signal as a function of just the state. As of 19/01/2023, this is a time step punishment of -1 for any state except for the MDP's terminal state.