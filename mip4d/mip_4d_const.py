import numpy as np

M1 = np.array([[1, 0, -1, 0], [0, 1, 0, -1]], dtype='int32')

B = np.zeros(shape=(2,12))
B[:,:4] = M1