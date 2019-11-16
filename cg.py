import numpy as np
np.random.seed( 0 )

n = 50
A = np.random.rand( n, n )
A += np.diag( np.linspace( n, 2 * n, n ) )
A += A.transpose()
B = np.zeros( n )
B[ 0 ] = 1

M = np.diag( 1 / np.diag( A ) )
# M = np.diag( np.ones( n ) )

# print( A )
# print( M )

 
w = np.zeros( n ) # init des poids

# un premier parcourt pour trouver le M, le z, mettre 0 dans p, et faire le produit scalaire h
r = B # résidu = 
z = M @ r 
p = z
for k in range( 10 ):
    # calcul de q + np.dot( p, q )
    q = A @ p

    # scalaires
    ha = np.dot( r, z )
    alpha = ha / np.dot( p, q )

    # maj w, r, z, h, err (vecteurs ?)
    w += alpha * p 
    r -= alpha * q 
    z = M @ r
    hb = np.dot( r, z )
    print( np.linalg.norm( r ) )

    # 
    beta = hb / ha
    p = z + beta * p

print( A @ w )
