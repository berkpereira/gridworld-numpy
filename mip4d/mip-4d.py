import numpy as np
import scipy.optimize

"""
For now, am trying to figure out how to get this whole formulation to make sense and work.
Will think of test cases and write functions along the way.
"""

max_altitude = 3
N = max_altitude

identity_stuff = np.zeros(shape=(4*N,4*(N+1)), dtype='int32')
for i in range(N):
    identity_stuff[4*i : 4*i + 4, 4*i : 4*i + 4] = -np.identity(4, dtype='int32')
    identity_stuff[4*i : 4*i + 4, 4*i + 4 : 4*i + 8] = np.identity(4, dtype='int32')

# define test direction history: down, right, right, up
d = np.array([1, 0, 0, 0,     0, 1, 0, 0,      0, 1, 0, 0,     0, 0, 1, 0], dtype='int32')

# calculate difference of velocities with linear method (Seb document, 2.3, first equation)
difference_d = np.matmul(identity_stuff, d)
# and then its L1 norm. objective is to minimise this.
l1_norm = np.linalg.norm(difference_d, ord=1)

# The below test functions create s and r as required
def create_s(difference_d):
    s = (difference_d > 0).astype(int)
    return s

def create_r(difference_d):
    r = - (difference_d < 0).astype(int)
    return r

# check for correction of s, r, as per Seb's document.
# However, I think we should also check for integers not larger in absolute value than 1!
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

s = create_s(difference_d)
r = create_r(difference_d)
s_padded = np.array(pad(s), dtype='int32')
r_padded = np.array(pad(r), dtype='int32')
u = create_u(s_padded, r_padded, d, N)
manhattan_helper = create_manhattan_helper(N)
print(np.matmul(manhattan_helper, u))