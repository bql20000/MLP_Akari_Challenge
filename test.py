import numpy as np

z = np.ones((5,5))
z[1][1] = 2
z[2][4] = 4
z[0][3] = 5
print(np.max(z, axis=0, keepdims=True))
z = np.exp(z - np.max(z, axis=0, keepdims=True))
print(z / np.sum(z, axis=0))


a = [[1,1,1],[-4,5,6],[7,-8,9]]
a = np.asarray(a)



