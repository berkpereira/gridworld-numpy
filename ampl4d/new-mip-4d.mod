param grid_size;
param T;
param initial_velocity_index;
param no_obstacles;
param initial {0..1};
param landing {0..1};
param obstacles{0..(no_obstacles-1), 0..1};


set TIME := 0..T; # set of discrete points in time. related to starting altitude.
set VELOCITY_INDEX := 0..3;
set POSITION_INDEX := 0..1;

# define variables
var Velocity {TIME, VELOCITY_INDEX} binary;
#var SlackPosLand {1..T, VELOCITY_INDEX} binary;
#var SlackNegLand {1..T, VELOCITY_INDEX} integer >= -1, <= 0;
var SlackPosLand {POSITION_INDEX} integer >= 0;
var SlackNegLand {POSITION_INDEX} integer <= 0;


# {x = s + r} constraint


# CONTINUE FROM HERE
# IN THE NEW VERSION, the slack variables are used to check the L-1 norm of the difference between the final state and the prescribed landing zone
#subj to SlackSum {t in 1..T, i in VELOCITY_INDEX} : Velocity[t,i] - Velocity[(t-1),i] = SlackPosLand[t,i] + SlackNegLand[t,i];
subj to SlackSum {i in POSITION_INDEX} : (sum {t in 0..(T-1)} (Velocity[t,i] - Velocity[t,i+2])) + initial[i] - landing[i] = SlackPosLand[i] + SlackNegLand[i];


# assign initial velocity as a constraint to the model.
# especially important for live closed-loop simulation, since in the midst of a simulation it can't just decide
# to begin the next part of its trajectory in any given velocity. Allowed turns would depend on its initial velocity.
subj to InitialVelocity : Velocity[0, initial_velocity_index] = 1;

# can't descend vertically nor travel diagonally.
# define constraint of valid velocities.
subj to SingleDirectionVelocity {t in TIME} : (sum {i in VELOCITY_INDEX} Velocity[t,i] = 1);
#subj to SingleDirectionVelocity {t in 0..(T-1)} : (sum {i in VELOCITY_INDEX} Velocity[t,i] = 1);


# prevent 180 degree turns
# see Seb's MIP encoding document, section 2.2, to understand the below: 
subj to NoBackTurnsOne {t in 1..T, i in 0..1} : Velocity[t-1,i] + Velocity[t,i+2] <= 1;
subj to NoBackTurnsTwo {t in 1..T, i in 2..3} : Velocity[t-1,i] + Velocity[t,i-2] <= 1;

# enforce boundaries for all steps in time, for both directions
# the below constraint with t in TIME was leading to some non-optimal solutions, so I CHANGED IT TO JUST t IN 1..(T-1)
#subj to Boundaries {t in TIME, j in 0..1} : 0 <= (sum {i in 0..(t-1)} (Velocity[i,j] - Velocity[i,j+2])) + initial[j] <= grid_size - 1 - initial[j];
subj to Boundaries {t in 0..(T-1), j in 0..1} : 0 <= (sum {i in 0..(t-1)} (Velocity[i,j] - Velocity[i,j+2])) + initial[j] <= grid_size - 1 - initial[j];

# other obstacles
#subj to ObstaclesX {t in TIME, j in OBSTACLES} : (sum {i in 0..t} VelocityXPlus[i] - VelocityXMinus[i]) + initialX < obstacles[i];

# finishing state constraint
# REMOVED IN THE NEW VERSION, INCLUDED IN OBJECTIVE INSTEAD
#subj to LandedState {j in POSITION_INDEX} : (sum {t in 0..(T-1)} (Velocity[t,j] - Velocity[t,j+2])) + initial[j] = landing[j];

# Objective: minimise number of turns
#minimize DirectionChanges : 0.5 * (sum {t in 1..T} ( (sum {i in VELOCITY_INDEX} SlackPosLand[t,i])
#	- (sum {i in VELOCITY_INDEX} SlackNegLand[t,i])));
minimize LandingError : sum {i in POSITION_INDEX} (SlackPosLand[i] - SlackNegLand[i]);