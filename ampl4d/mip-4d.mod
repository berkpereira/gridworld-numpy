param grid_size;
param T;
param initial {0..1};
param landing {0..1};
#param obstacles;

set   TIME := 0..T; # set of discrete points in time. related to starting altitude.
set VELOCITY_INDEX := 0..3;

# define variables
var Velocity {TIME, VELOCITY_INDEX} binary;
var SlackPos {1..T, VELOCITY_INDEX} binary;
var SlackNeg {1..T, VELOCITY_INDEX} integer >= -1, <= 0;

# {x = s + r} constraint
subj to SlackSum {t in 1..T, i in VELOCITY_INDEX} : Velocity[t,i] - Velocity[(t-1),i] = SlackPos[t,i] + SlackNeg[t,i];

# can't descend vertically nor travel diagonally.
# define constraint of valid velocities.
subj to SingleDirectionVelocity {t in TIME} : sum {i in VELOCITY_INDEX} Velocity[t,i] = 1;

# prevent 180 degree turns
# see Seb's MIP encoding document, section 2.2, to understand the below: 
subj to NoBackTurnsOne {t in 1..T, i in 0..1} : Velocity[t-1,i] + Velocity[t,i+2] <= 1;
subj to NoBackTurnsTwo {t in 1..T, i in 2..3} : Velocity[t-1,i] + Velocity[t,i-2] <= 1;

# enforce boundaries for all steps in time, for both directions
subj to Boundaries {t in TIME, j in 0..1} : 0 <= (sum {i in 0..(t-1)} (Velocity[i,j] - Velocity[i,j+2])) + initial[j] <= grid_size - 1 - initial[j];

# other obstacles
#subj to ObstaclesX {t in TIME, j in OBSTACLES} : (sum {i in 0..t} VelocityXPlus[i] - VelocityXMinus[i]) + initialX < obstacles[i];

# finishing state constraint
subj to LandedState {j in 0..1} : (sum {t in 0..(T-1)} (Velocity[t,j] - Velocity[t,j+2])) + initial[j] = landing[j];

# Objective: minimise number of turns
minimize DirectionChanges : 0.5 * (sum {t in 1..T} ( (sum {i in VELOCITY_INDEX} SlackPos[t,i])
	- (sum {i in VELOCITY_INDEX} SlackNeg[t,i])));