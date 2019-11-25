# # for nm in [ 7, 8 ]:
# #     res = 2
# #     for nb_points in range( 3, nm + 1 ):
# #         t = 1
# #         for j in range( 2, nb_points ):
# #             t *= j
# #         print( nb_points, t )
# #         res += t * ( 2**nb_points - 2 )



# #     print( nm, res )
# from pysdot.domain_types import ConvexPolyhedraAssembly
# from pysdot.radial_funcs import RadialFuncInBall
# from pysdot import OptimalTransport
# import numpy as np

# p = np.array( [
#     [ 7.1831867 ,  4.80071334],
#     [25.03284266,  1.8461205 ],
#     [ 2.91926123,  2.27110824],
#     [25.10334747,  4.57506844],
#     [ 7.65621233,  7.71841629],
#     [22.19798178,  3.05121635],
#     [24.3487165 ,  0.56329483],
#     [25.32667855,  0.49470554],
#     [ 8.70768298,  1.39365694],
#     [14.42269486,  2.14316306],
#     [ 3.91237168,  0.90472185],
#     [ 4.56647449,  2.22325867],
#     [25.56699626,  9.19262079],
#     [ 1.93584725,  5.40965525],
#     [ 1.92642132,  0.93346305],
#     [26.22040181,  0.63560641]
# ] )
# N = p.shape[ 0 ]

# domain = ConvexPolyhedraAssembly()
# domain.add_box([0,0], [10, 10])
# domain.add_box([10, 4], [18, 4.6])
# domain.add_box([18,0],[28,10])

# eps_density=2
# domain.add_box([10,0], [18,4], coeff=eps_density)
# domain.add_box([10,4.6], [18,10], coeff=eps_density)


# # diracs
# ot = OptimalTransport( 
#     positions = p, 
#     weights = np.ones( N ) * 1e-0, 
#     masses = np.ones( N ) * 1e-0, 
#     domain = domain, 
#     radial_func = RadialFuncInBall()
# )
# ot.verbosity = 1

# # solve
# ot.adjust_weights()

# print( ot.pd.integrals() )

# # display
# ot.display_vtk( "results/pd.vtk" )
import matplotlib.pylab as plt
import numpy as np

a= [
    (0, 0),
    (1, 0),
    (2, 0),
    (3, 0.13823),
    (4, 0.231548),
    (5, 0.242912),
    (6, 0.18745),
    (7, 0.114503),
    (8, 0.0546218),
    (9, 0.0218455),
    (10, 0.00646962),
    (11, 0.00192397),
    (12, 0.000417565),
    (13, 6.87133e-05),
    (14, 1.05713e-05)
]

x = np.array( [ x[ 0 ] for x in a ] )
y = np.array( [ x[ 1 ] for x in a ] )

print( sum( y[ 9: ] ) )

plt.plot( x, y )
plt.show()
