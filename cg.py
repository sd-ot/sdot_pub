import numpy as np

n = 5
A = np.random.rand( n, n )
A += np.diag( n * np.ones( n ) )
A += A.transpose()
b = np.ones( n )

M = np.diag( 1 / np.diag( A ) )

print( A )
print( M )

x = np.zeros( n )
r = b
p = r
# z = 