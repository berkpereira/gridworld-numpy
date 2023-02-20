import numpy as np
import scipy.optimize


# Input constraint

backward_turn_matrix = np.array([[1,0,0,0,0,0,1,0], [0,1,0,0,0,0,0,1], [0,0,1,0,1,0,0,0], [0,0,0,1,0,1,0,0]], dtype='int32')

print(backward_turn_matrix)