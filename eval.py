# # for nm in [ 7, 8 ]:
# #     res = 2
# #     for nb_points in range( 3, nm + 1 ):
# #         t = 1
# #         for j in range( 2, nb_points ):
# #             t *= j
# #         print( nb_points, t )
# #         res += t * ( 2**nb_points - 2 )



# #     print( nm, res )
from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall
from pysdot import OptimalTransport
import numpy as np

p = np.array( [
    [ 9.76839859,  6.88125405],
    [ 8.04386812,  6.63015642],
    [19.20456767,  6.71223903],
    [21.1511015 ,  1.29290147],
    [21.14421864,  5.39551101],
    [20.8086977 ,  3.6997252 ],
    [11.84435201,  4.94440997],
    [21.12562426,  7.12708042],
    [18.213685  ,  6.68852209],
    [10.56343447,  4.38092558],
    [22.26572103,  4.03077149],
    [ 6.52951969,  4.95031689],
    [10.45120119,  6.60141607],
    [20.9421186 ,  6.50181466],
    [ 8.67406991,  1.36978849],
    [14.49477989,  5.97103317]
] )
N = p.shape[ 0 ]

domain = ConvexPolyhedraAssembly()
domain.add_box([0,0], [10, 10])
domain.add_box([10, 4], [18, 4.6])
domain.add_box([18,0],[28,10])
eps_density=0.01
domain.add_box(min_pos=[10,0], max_pos=[18,4], coeff=eps_density)
domain.add_box(min_pos=[10,4.6], max_pos=[18,10], coeff=eps_density)
point_infinity_1 = np.array([22, 2])
point_infinity_2=np.array([22,8])

# diracs
ot = OptimalTransport( 
    positions = p, 
    weights = np.ones( N ),
    masses = np.ones( N ) * 10 / 16,
    domain = domain, 
    radial_func = RadialFuncInBall()
)
ot.verbosity = 1

# solve
ot.adjust_weights( relax = 0.5 )

print( ot.pd.integrals() )

# # # display
ot.display_vtk( "results/pd.vtk" )
# import matplotlib.pylab as plt
# import numpy as np

# # a= [
# #     (0, 0),
# #     (1, 0),
# #     (2, 0),
# #     (3, 0.13823),
# #     (4, 0.231548),
# #     (5, 0.242912),
# #     (6, 0.18745),
# #     (7, 0.114503),
# #     (8, 0.0546218),
# #     (9, 0.0218455),
# #     (10, 0.00646962),
# #     (11, 0.00192397),
# #     (12, 0.000417565),
# #     (13, 6.87133e-05),
# #     (14, 1.05713e-05)
# # ]

# a= [
#     (0, 0),
#     (1, 0),
#     (2, 0),
#     (3, 0),
#     (4, 0.00016),
#     (5, 0.00152),
#     (6, 0.04032),
#     (7, 0.23528),
#     (8, 0.35776),
#     (9, 0.24144),
#     (10, 0.08816),
#     (11, 0.028),
#     (12, 0.00616),
#     (13, 0.00104),
#     (14, 0.00016)
# ]

# x = np.array( [ x[ 0 ] for x in a ] )
# y = np.array( [ x[ 1 ] for x in a ] )

# print( sum( y[ 9: ] ) )

# plt.plot( x,  np.cumsum( y ) )
# plt.show()
