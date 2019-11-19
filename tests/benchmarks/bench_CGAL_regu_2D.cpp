//// nsmake avoid_inc CGAL/

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O5
//// nsmake lib_flag -O5

//// nsmake lib_path /usr/local/lib
//// nsmake lib_name CGAL
//// nsmake lib_name gmp

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Regular_triangulation_2.h>
#include "../../src/sdot/Support/Time.h"
#include <fstream>
#include <cnpy.h>

using K = CGAL::Exact_predicates_inexact_constructions_kernel;
using R = CGAL::Regular_triangulation_2<K>;

void f( const char *filename ) {

    // load
    cnpy::NpyArray arr = cnpy::npy_load( filename );
    std::size_t nb_diracs = arr.shape[ 1 ];
    double *data = arr.data<double>();

    std::vector<R::Weighted_point> diracs( nb_diracs );
    for( std::size_t n = 0; n < nb_diracs; ++n )
        diracs[ n ] = { { data[ n + 0 * nb_diracs ], data[ n + 1 * nb_diracs ] }, data[ n + 2 * nb_diracs ] };

    // exec
    std::uint64_t t0 = 0, t1 = 0, nb_reps = 1;
    RDTSC_START( t0 );

    R rt( diracs.begin(), diracs.end() );

    //    std::string cmd = "cat /proc/" + std::to_string( getpid() ) + "/maps";
    //    system( cmd.c_str() );

    double s = 0;
    for( auto v = rt.all_vertices_begin(); v != rt.all_vertices_end(); ++v ) {
        auto circulator = rt.incident_faces( v ), done( circulator );
        do {
            double v = circulator->vertex( 0 )->point().point().x();
            s += v;
        } while( ++circulator != done );
    }

    RDTSC_FINAL( t1 );
    std::cout << nb_diracs << " " << double( t1 - t0 ) / nb_diracs / nb_reps << std::endl;
}

int main() {
    const char *filenames[] = {
        "/data/sdot/uniform_100000_2D_solved.npy",
        "/data/sdot/uniform_200000_2D_solved.npy",
        "/data/sdot/uniform_400000_2D_solved.npy",
        "/data/sdot/uniform_800000_2D_solved.npy",
        "/data/sdot/uniform_1600000_2D_solved.npy",
    };

    for( const char *filename : filenames )
        f( filename );

    return 0;
}
