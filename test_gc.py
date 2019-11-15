import numpy as np

n = 5
A = np.random.rand( n, n )
A += np.diag( n + np.random.rand( n ) )
A += A.transpose()
print( A )

M


r = b 
z = 