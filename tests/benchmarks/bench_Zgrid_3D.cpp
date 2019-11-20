#include "../../src/sdot/Support/StaticRange.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
#include "../../src/sdot/Grids/ZGrid.h"
#include <fstream>
#include <cnpy.h>
#include <map>
using namespace sdot;

// // nsmake cxx_name clang++
// // nsmake cpp_flag -march=skylake

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O3
//// nsmake lib_name z

template<int _dim>
struct Pc {
    enum { store_the_normals = false };
    enum { allow_ball_cut    = false };
    enum { dim               = _dim };
    using  TF                = double;
    using  TI                = std::size_t;
    struct Dirac {};
};

template<class Pc,int voronoi>
void test( std::map<std::size_t,double> &nb_cycle_per_cell, std::string filename, N<voronoi>, bool dot_npy = false ) {
    constexpr int dim = Pc::dim;
    using Grid = ZGrid<Pc>;

    using CP = typename Grid::CP;
    using Pt = typename Grid::Pt;
    using TF = typename Grid::TF;
    using TI = typename Grid::TI;

    // load
    std::array<const TF *,dim> positions;
    std::vector<TF> tmp[ dim + 1 ];
    std::size_t nb_diracs;
    double *weights;
    if ( dot_npy ) {
        cnpy::NpyArray arr = cnpy::npy_load( filename );
        nb_diracs = arr.shape[ 1 ];
        TF *data = arr.data<TF>();

        for( std::size_t i = 0; i < dim; ++i )
            positions[ i ] = data + i * nb_diracs;
        weights = data + dim * nb_diracs;
    } else {
        std::ifstream fin( filename );
        while ( true ) {
            TF p[ dim ], w;
            for( std::size_t d = 0; d < dim; ++d )
                fin >> p[ d ];
            fin >> w;
            if ( ! fin )
                break;
            for( std::size_t d = 0; d < dim; ++d )
                tmp[ d ].push_back( p[ d ] );
            tmp[ dim ].push_back( w * 0 );
        }
        for( std::size_t d = 0; d < dim; ++d )
            positions[ d ] = tmp[ d ].data();
        weights = tmp[ dim ].data();
        nb_diracs = tmp[ 0 ].size();
    }


    // get timings
    double best_dt_sum = 1e6, smurf = 0;
    for( std::size_t nb_diracs_per_cell = 53; nb_diracs_per_cell <= 53; nb_diracs_per_cell += 2 ) {
        constexpr int flags = Grid::homogeneous_weights * voronoi;
        // RaiiTime re("total");

        std::uint64_t t0_grid = 0, t1_grid = 0;
        RDTSC_START( t0_grid );
        Grid grid( nb_diracs_per_cell );
        grid.update( positions, weights, nb_diracs, N<flags>() );
        RDTSC_FINAL( t1_grid );

        CP b( typename CP::Box{ { 0, 0 }, { 1, 1 } } );
        std::vector<std::size_t> nb_cuts( 16 * thread_pool.nb_threads(), 0 );
        std::uint64_t t0_each = 0, t1_each = 0;
        RDTSC_START( t0_each );
        grid.for_each_laguerre_cell( [&]( CP &cp, std::size_t /*num*/, int num_thread ) {
            nb_cuts[ 16 * num_thread ] += cp.nb_nodes();
        }, b, positions, weights, nb_diracs, N<flags>() );
        RDTSC_FINAL( t1_each );

        // double dt_grid = double( t1_grid - t0_grid ) / nb_diracs;
        // double dt_each = double( t1_each - t0_each ) / nb_diracs;
        double dt_sum  = double( t1_each - t0_grid ) / nb_diracs;
        // P( nb_diracs_per_cell, dt_sum );
        best_dt_sum = std::min( best_dt_sum, dt_sum );
        smurf += nb_cuts[ 0 ];
    }
    P( nb_diracs, smurf, best_dt_sum );

    nb_cycle_per_cell[ nb_diracs ] = best_dt_sum;

    //    grid.display_tikz( std::cout, 10.0 );
    //    for( std::size_t i = 0; i < positions.size(); ++i )
    //        std::cout << "\\draw[blue] (" << 10 * positions[ i ].x << "," << 10 * positions[ i ].y << ") node {$\\times$};\n";

    //    VtkOutput vo;
    //    grid.display( vo );
    //    vo.save( "vtk/grid.vtk" );

}

int main() {
    const char *filenames[] = {
        "/data/sdot/faces_20p_2D_1000000.txt",
        "/data/sdot/faces_20p_2D_1600000.txt",
        "/data/sdot/faces_20p_2D_2560000.txt",
        "/data/sdot/faces_20p_2D_4096000.txt",
        "/data/sdot/faces_20p_2D_6553600.txt",
        "/data/sdot/faces_20p_2D_10485760.txt",
        "/data/sdot/faces_20p_2D_16777216.txt",
        "/data/sdot/faces_20p_2D_26843545.txt",
        //        "/data/sdot/faces_20p_2D_42949672.txt",
        //        "/data/sdot/faces_20p_2D_68719475.txt",
        //        "/data/sdot/faces_20p_2D_109951160.txt",
        //        "/data/sdot/faces_20p_2D_175921856.txt",
        //        "/data/sdot/faces_20p_2D_281474969.txt",
    };

    std::map<std::size_t,double> nb_cycle_per_cell;
    for( const char *filename : filenames )
        test<Pc<2>>( nb_cycle_per_cell, filename, N<true>() );

    std::cout << "    \\addplot coordinates {\n";
    for( auto p : nb_cycle_per_cell )
        std::cout << "       ( " << p.first << ", " << p.second << " )\n";
    std::cout << "    };\n";
}
