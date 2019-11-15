import numpy as np

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

#
be = 0

# un premier parcourt pour trouver le M, le z, mettre 0 dans p, et faire le produit scalaire h
r = B # résidu = 
p = 0 
z = M @ r 
h = np.dot( r, z )

for k in range( 10 ):
    # mise à jour de p, calcul de chaque terme de q, puis produit scalaire
    p = z + be * p
    q = A @ p
    alpha = h / np.dot( p, q )

    # maj w et r, calcul de norme de r
    o = h 
    w += alpha * p 
    r -= alpha * q 

    z = M @ r
    h = np.dot( r, z )
    print( np.linalg.norm( r ) )

    # 
    be = h / o 

print( A @ w )
