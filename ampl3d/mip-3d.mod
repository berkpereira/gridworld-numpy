param grid_size;
param T;
param initial {0..1};
param landing {0..1};


set TIME := 0..T; # set of discrete points in time. related to starting altitude.
set VELOCITY_INDEX := 0..3;
set POSITION_INDEX := 0..1;

# define variables
var Velocity {TIME, VELOCITY_INDEX} binary;
var SlackPosLand {POSITION_INDEX} integer >= 0;
var SlackNegLand {POSITION_INDEX} integer <= 0;

# {x = s + r} constraint
#subj to SlackSum {t in 1..T, i in VELOCITY_INDEX} : Velocity[t,i] - Velocity[(t-1),i] = SlackPosLand[t,i] + SlackNegLand[t,i];
subj to SlackSumLand {i in POSITION_INDEX} : (sum {t in 0..(T-1)} (Velocity[t,i] - Velocity[t,i+2])) + initial[i] - landing[i] = SlackPosLand[i] + SlackNegLand[i];

# can't travel diagonally (but can now do it vertically!)
# define constraint of valid velocities.
subj to SingleDirectionVelocity {t in TIME} : sum {i in VELOCITY_INDEX} Velocity[t,i] <= 1;

# enforce boundaries for all steps in time, for both directions
subj to Boundaries {t in TIME, j in 0..1} : 0 <= (sum {i in 0..(t-1)} (Velocity[i,j] - Velocity[i,j+2])) + initial[j] <= grid_size - 1;

# other obstacles
#subj to ObstaclesX {t in TIME, j in OBSTACLES} : (sum {i in 0..t} VelocityXPlus[i] - VelocityXMinus[i]) + initialX < obstacles[i];

# finishing state constraint
#subj to LandedState {j in 0..1} : (sum {t in 0..(T-1)} (Velocity[t,j] - Velocity[t,j+2])) + initial[j] = landing[j];

# Objective: minimise number of turns
#minimize DirectionChanges : 0.5 * (sum {t in 1..T} ( (sum {i in VELOCITY_INDEX} SlackPosLand[t,i])
#	- (sum {i in VELOCITY_INDEX} SlackNegLand[t,i])));
minimize LandingError : sum {i in POSITION_INDEX} (SlackPosLand[i] - SlackNegLand[i]);