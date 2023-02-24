import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
import mip_4d_const as const
import time

"""
For now, am trying to figure out how to get this whole formulation to make sense and work.
Will think of test cases and write functions along the way.

In this context, N is equal to the max altitude
"""

max_altitude = 3
N = max_altitude - 1

# we need to add extra zero rows to accommodate the slack variables
def create_input_constraint(N):
    identity_stuff = np.zeros(shape=(4*N,4*(N+1)), dtype='int32')
    for i in range(N):
        identity_stuff[4*i : 4*i + 4, 4*i : 4*i + 4] = -np.identity(4, dtype='int32')
        identity_stuff[4*i : 4*i + 4, 4*i + 4 : 4*i + 8] = np.identity(4, dtype='int32')
    
    # this is REALLY slow. copying arrays a bunch of times
    identity_stuff = np.insert(identity_stuff, np.repeat(np.arange(4, 4*N + 1, 4), 8), 0, axis=1)
    # ALSO append to the end as well!
    identity_stuff = np.append(identity_stuff, np.zeros(shape=(4*N, 8)), axis=1).astype('int32')

    # now create bounds on the constraint
    #lower_bounds = np.zeros(shape = 4*N)
    upper_bounds =  np.ones(shape = 4*N)

    return LinearConstraint(identity_stuff, lb=-np.inf, ub=upper_bounds)

# define test direction history: down, right, right, up
#d = np.array([1, 0, 0, 0,     0, 1, 0, 0,      0, 1, 0, 0,     0, 0, 1, 0], dtype='int32')

# calculate difference of velocities with linear method (Seb document, 2.3, first equation)
#difference_d = np.matmul(identity_stuff, d)
# and then its L1 norm. objective is to minimise this.
#l1_norm = np.linalg.norm(difference_d, ord=1)

# The below test functions create s and r as required
def create_s(difference_d):
    s = (difference_d > 0).astype(int)
    return s

def create_r(difference_d):
    r = - (difference_d < 0).astype(int)
    return r

# check for correction of s, r, as per Seb's document.
# However, I think we should also check for integers not larger in absolute value than 1!
# thus, all decision variables are constrained to be integers
def is_s_r_correct(s, r, difference_d):
    if not np.array_equal(s + r, difference_d):
        return False
    if not (s >= 0).all():
        return False
    if not (r <= 0).all():
        return False
    return True

# pad s or r
def pad(s):
    padded = np.zeros(shape=s.size + 4)
    padded[:s.size] = s
    return padded

def create_u(s_padded,r_padded,d, N):
    u = np.zeros(shape=12*(N+1))
    for i in range(N + 1):
        u_k = np.zeros(shape=12)
        u_k[:4] = d[4*i:4*i+4]
        u_k[4:8] = s_padded[4*i:4*i+4]
        u_k[8:12] = r_padded[4*i:4*i+4]
        u[12*i:12*i+12] = u_k
    return u

def create_manhattan_helper(N):
    manhattan_helper = np.zeros(shape=12* (N + 1))
    manhattan_instance = np.zeros(shape=12)
    manhattan_instance[4:8] = 1
    manhattan_instance[8:12] = -1
    for i in range(N + 1):
        manhattan_helper[12*i:12*i + 12] = manhattan_instance
    return manhattan_helper

def create_M2(N):
    return np.tile(const.B, N + 1)

def create_u_bounds(N):
    lower_bounds = np.zeros(shape=12*(N + 1))
    upper_bounds = np.zeros(shape=12*(N + 1))
    for i in range(upper_bounds.size):
        mod = i % 12
        if mod < 4: # bounds on d_k
            lower_bounds[i] = 0
            upper_bounds[i] = 1
        elif mod < 8: # bounds on s_k
            lower_bounds[i] = 0
            upper_bounds[i] = 1
        else: # bounds on r_k
            lower_bounds[i] = -1
            upper_bounds[i] = 0
    return Bounds(lb=lower_bounds, ub=upper_bounds)

# matrix used in 2nd line of canonical scipy form
def create_A_constraint(N, terminal_state):
    M2 = create_M2(N)
    A = np.zeros(shape=(2 + N + 1, 12*(N+1)))
    A[:2, :] = M2
    
    # fill in extra rows for constraining d_k sums
    for i in range(2, A.shape[0]):
        A[i, 12*(i - 2):12*(i - 2)+4] = 1
    lower_bounds, upper_bounds = create_Au_bounds(terminal_state, N, M2)

    return LinearConstraint(A, lb=lower_bounds, ub=upper_bounds)

# bounds in 2nd line of canonical scipy form
# at this moment, assuming starting state of [0,0]
def create_Au_bounds(terminal_state, N, M2):
    #lower_bounds = np.zeros(shape=12*(N+1) + N + 1)
    #upper_bounds = np.zeros(shape=12*(N+1) + N + 1)
    
    lower_bounds = np.zeros(shape= 2 + N + 1)
    lower_bounds[:2] = terminal_state
    lower_bounds[2:] = 1
    upper_bounds = np.copy(lower_bounds)

    return lower_bounds, upper_bounds

def create_int_constraint(N):
    return np.ones(shape=12*(N+1))

def extract_d_variables(u, N):
    d_variables = np.zeros(shape=4*(N+1))
    j = 0
    for i in range(u.size):
        if i % 12 < 4:
            d_variables[j] = u[i]
            j += 1
    return d_variables.astype('int32')


# REMEMBER: N = max altitude - 1
def mip_optimise(N, terminal_state):
    A_constraint = create_A_constraint(N, terminal_state)
    u_bounds = create_u_bounds(N)
    int_constraint = create_int_constraint(N)
    objective_coeffs = create_manhattan_helper(N)
    input_constraint = create_input_constraint(N)
    constraints = [input_constraint, A_constraint]
    #constraints = A_constraint

    result = milp(c=objective_coeffs, integrality=int_constraint, bounds=u_bounds, constraints=constraints, options={'disp':True})
    return result


if __name__ == '__main__':
    max_altitude = 1000
    N = max_altitude - 1
    terminal_state = np.array([1000,0], dtype='int32')
    st = time.time()
    solution = mip_optimise(N, terminal_state)
    et = time.time()
    print('Done')
    print(f'Time elapsed: {et - st} seconds')
    if solution.x is None:
        print('No solution found')
    else:
        print('Solution found.')
        print('d variable history below:')
        print(extract_d_variables(solution.x, N))
