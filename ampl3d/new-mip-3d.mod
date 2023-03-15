param grid_size;
param T;
#param initial_velocity_index;
param no_obstacles;
param initial {0..1};
param landing {0..1};
param obstacles {0..(no_obstacles-1), 0..1};
param example_obstacle {0..1};


set TIME := 0..T; # set of discrete points in time. related to starting altitude.
set VELOCITY_INDEX := 0..3;
set POSITION_INDEX := 0..1;
set OBSTACLE_INDEX := 0..(no_obstacles-1);

# define variables
var Velocity {TIME, VELOCITY_INDEX} binary;
var SlackPosLand {POSITION_INDEX} integer >= 0;
var SlackNegLand {POSITION_INDEX} integer <= 0;

# as per Richards2002 paper:
#var ObstacleRelax {TIME, OBSTACLE_INDEX, VELOCITY_INDEX, POSITION_INDEX} binary;
var ObstacleRelax {TIME, OBSTACLE_INDEX, VELOCITY_INDEX} binary;

# cannot relax more than 3 directions. As per Richards2002 paper:
subj to ObstacleRelaxLimit {k in TIME, i in OBSTACLE_INDEX} : (sum {v in VELOCITY_INDEX} (ObstacleRelax[k,i,v])) <= 3;

# avoid obstacles 
subj to AvoidObstacleRowPos {k in TIME, i in OBSTACLE_INDEX} :   (sum {t in 0..(k-1)} (Velocity[t,0] - Velocity[t,2])) + initial[0] - obstacles[i,0] >= 1 - (10000 * ObstacleRelax[k,i,0]);
subj to AvoidObstacleRowNeg {k in TIME, i in OBSTACLE_INDEX} : - (sum {t in 0..(k-1)} (Velocity[t,0] - Velocity[t,2])) - initial[0] + obstacles[i,0] >= 1 - (10000 * ObstacleRelax[k,i,1]);
subj to AvoidObstacleColPos {k in TIME, i in OBSTACLE_INDEX} :   (sum {t in 0..(k-1)} (Velocity[t,1] - Velocity[t,3])) + initial[1] - obstacles[i,1] >= 1 - (10000 * ObstacleRelax[k,i,2]);
subj to AvoidObstacleColNeg {k in TIME, i in OBSTACLE_INDEX} : - (sum {t in 0..(k-1)} (Velocity[t,1] - Velocity[t,3])) - initial[1] + obstacles[i,1] >= 1 - (10000 * ObstacleRelax[k,i,3]);

# {x = s + r} slack variables constraint
# IN THE NEW VERSION, the slack variables are used to check the L-1 norm of the difference between the final state and the prescribed landing zone
#subj to SlackSum {t in 1..T, i in VELOCITY_INDEX} : Velocity[t,i] - Velocity[(t-1),i] = SlackPosLand[t,i] + SlackNegLand[t,i];
subj to SlackSumLand {i in POSITION_INDEX} : (sum {t in 0..(T-1)} (Velocity[t,i] - Velocity[t,i+2])) + initial[i] - landing[i] = SlackPosLand[i] + SlackNegLand[i];

# assign initial velocity as a constraint to the model.
# especially important for live closed-loop simulation, since in the midst of a simulation it can't just decide
# to begin the next part of its trajectory in any given velocity. Allowed turns would depend on its initial velocity.
#subj to InitialVelocity : Velocity[0, initial_velocity_index] = 1;

# can't travel diagonally.
# define constraint of valid velocities.
subj to SingleDirectionVelocity {t in TIME} : sum {i in VELOCITY_INDEX} (Velocity[t,i]) <= 1;

# prevent 180 degree turns
# see Seb's MIP encoding document, section 2.2, to understand the below: 
#subj to NoBackTurnsOne {t in 1..T, i in 0..1} : Velocity[t-1,i] + Velocity[t,i+2] <= 1;
#subj to NoBackTurnsTwo {t in 1..T, i in 2..3} : Velocity[t-1,i] + Velocity[t,i-2] <= 1;

# enforce boundaries for all steps in time, for both directions
# the below constraint with t in TIME was leading to some non-optimal solutions, so I CHANGED IT TO JUST t IN 1..(T-1)
subj to Boundaries {t in TIME, j in POSITION_INDEX} : 0 <= (sum {i in 0..(t-1)} (Velocity[i,j] - Velocity[i,j+2])) + initial[j] <= grid_size - 1;


# Objective: minimise number of turns
#minimize DirectionChanges : 0.5 * (sum {t in 1..T} ( (sum {i in VELOCITY_INDEX} SlackPosLand[t,i])
#	- (sum {i in VELOCITY_INDEX} SlackNegLand[t,i])));

# NEW OBJECTIVE:
# finish as close as possible to the prescribed landing zone.
minimize LandingError : sum {i in POSITION_INDEX} (SlackPosLand[i] - SlackNegLand[i]);