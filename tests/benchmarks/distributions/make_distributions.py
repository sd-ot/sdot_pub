from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import OptimalTransport
from matplotlib import pyplot as plt
import numpy as np
import os, sys

def mass( simplex ):
    if len( simplex ) == 4:
        dx = simplex[ 2 ] - simplex[ 0 ]
        dy = simplex[ 3 ] - simplex[ 1 ]
        return np.sqrt( dx ** 2 + dy ** 2 )

    if len( simplex ) == 6:
        raise "TODO"

    raise "TODO"

def point_on_simplex( simplex, sd ):
    if len( simplex ) == 4:
        x0 = simplex[ 0 ]
        y0 = simplex[ 1 ]
        dx = simplex[ 2 ] - x0
        dy = simplex[ 3 ] - y0
        return [ x0 + sd * dx, y0 + sd * dy ]

    if len( simplex ) == 6:
        raise "TODO"

    raise "TODO"

def random_point_on_simplex( simplex, dr = 1e-3 ):
    if len( simplex ) == 4:
        p = point_on_simplex( simplex, np.random.rand() )
        xd = dr * np.random.normal() * 0 
        yd = dr * np.random.normal() * 0
        return [ p[ 0 ] + xd, p[ 1 ] + yd ]

    if len( simplex ) == 6:
        raise "TODO"

    raise "TODO"

def make_positions( dist_name, nb_diracs, dim ):
    if dist_name == "uniform":
        return np.random.rand( nb_diracs, dim )

    if dist_name.startswith( "voro_" ):
        # gen simplex list
        gen = os.path.join( os.path.dirname( sys.argv[ 0 ] ), "make_simplex_list_voro_bounds.cpp" )
        out = "/data/sdot/{}_{}D".format( dist_name, dim )
        if os.system( "nsmake run '{}' {} {} {}".format( gen, out, dist_name.split( "_" )[ 1 ], dim ) ):
            sys.exit( 1 )

        simplex_list = []
        for l in open( out ).readlines():
            simplex_list.append( list( map( float, l.split() ) ) )

        # masses
        acc = 0
        acc_masses = []
        for simplex in simplex_list:
            acc += mass( simplex )
            acc_masses.append( acc )
        for i in range( len( acc_masses ) ):
            acc_masses[ i ] /= acc

        # nb_points_per_simplex
        num_simplex = 0
        nb_points_per_simplex = np.zeros( len( simplex_list ) )
        for n in range( nb_diracs ):
            a = ( n + 0.5 ) / nb_diracs
            while acc_masses[ num_simplex ] < a:
                num_simplex += 1
            nb_points_per_simplex[ num_simplex ] += 1

        # points
        num_dirac = 0
        res = np.zeros( [ nb_diracs, dim ] )
        for num_simplex in range( len( simplex_list ) ):
            nb_points = int( nb_points_per_simplex[ num_simplex ] )
            for n in range( nb_points ):
                # res[ n, : ] = random_point_on_simplex( simplex_list[ num_simplex ] )
                res[ num_dirac, : ] = point_on_simplex( simplex_list[ num_simplex ], ( n + 0.5 ) / nb_points )
                res[ num_dirac, : ] += 1 / nb_diracs * np.array( [ np.random.normal(), np.random.normal() ] )
                num_dirac += 1

        # plt.plot( res[ :, 0 ], res[ :, 1 ], "x" )
        # plt.show()
        return res


for dist_name in [ "uniform" ]: # , "voro_50"
    for dim in [ 2, 3 ]:
        for nb_diracs in map( int, [ 1e5, 2e5, 4e5, 8e5, 16e5 ] ): # 
            positions = make_positions( dist_name, nb_diracs, dim )

            # diracs
            ot = OptimalTransport( positions = positions )
            ot.verbosity = 1

            # solve
            print( dim, nb_diracs )
            ot.adjust_weights()

            # display, save
            if nb_diracs <= 1e6:
                ot.display_vtk( "vtk/pd.vtk", points = True )

            pw = np.zeros( [ dim + 1, nb_diracs ] )
            for d in range( dim ):
                pw[ d, : ] = positions[ :, d ]
            np.save( "/data/sdot/{}_{}_{}D_equalw.npy".format( dist_name, nb_diracs, dim ), pw )

            pw[ dim, : ] = ot.get_weights()
            np.save( "/data/sdot/{}_{}_{}D_solved.npy".format( dist_name, nb_diracs, dim ), pw )
