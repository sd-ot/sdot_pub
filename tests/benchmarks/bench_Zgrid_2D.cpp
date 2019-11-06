#include "../../src/sdot/Support/StaticRange.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
#include "../../src/sdot/ZGrid.h"
#include <cnpy.h>
#include <map>
using namespace sdot;

// // nsmake cxx_name clang++
// // nsmake cpp_flag -ffast-math
// // nsmake cpp_flag -march=skylake

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O2
//// nsmake lib_name z

template<int _dim>
struct Pc {
    enum { store_the_normals = false };
    enum { allow_ball_cut    = false };
    enum { dim               = _dim };
    using  TF                = double;
    using  TI                = std::size_t;
    using  CI                = std::size_t;
};

//template<class TF>
//void test_vol( const std::vector<TF> &positions_x, const std::vector<TF> &positions_y, const std::vector<TF> &weights ) {
//    using Grid = ZGrid<Pc>;
//    using CP = Grid::CP;

//    Grid grid( 10 );
//    grid.update( { positions_x.data(), positions_y.data() }, weights.data(), weights.size(), N<Grid::homogeneous_weights>() );

//    CP b( CP::Box{ { 0, 0 }, { 1, 1 } } );
//    std::vector<TF> vols( thread_pool.nb_threads(), 0 );
//    grid.for_each_laguerre_cell( [&]( CP &cp, std::size_t /*num*/, int num_thread ) {
//        vols[ num_thread ] += cp.integral();
//    }, b, { positions_x.data(), positions_y.data() }, weights.data(), weights.size(), N<Grid::homogeneous_weights>() );

//    TF vol = 0;
//    for( TF v : vols )
//        vol += v;
//    P( vol );
//}

template<class Pc,int voronoi>
void test( std::map<std::size_t,double> &nb_cycle_per_cell, std::string filename, N<voronoi> ) {
    constexpr int dim = Pc::dim;
    using Grid = ZGrid<Pc>;
    using CP = typename Grid::CP;
    using Pt = typename Grid::Pt;
    using TF = typename Grid::TF;
    using TI = typename Grid::TI;

    // load
    cnpy::NpyArray arr = cnpy::npy_load( filename );
    std::size_t nb_diracs = arr.shape[ 1 ];
    double *data = arr.data<double>();

    std::array<const double *,dim> positions;
    for( std::size_t i = 0; i < dim; ++i )
        positions[ i ] = data + i * nb_diracs;
    double *weights = data + dim * nb_diracs;

    // get timings
    double best_dt_sum = 1e6, smurf = 0;
    for( std::size_t nb_diracs_per_cell = 23; nb_diracs_per_cell <= 33; nb_diracs_per_cell += 1 ) {
        constexpr int flags = Grid::homogeneous_weights * voronoi;
        // RaiiTime re("total");

        std::uint64_t t0_grid = 0, t1_grid = 0;
        RDTSC_START( t0_grid );
        Grid grid( nb_diracs_per_cell );
        grid.update( { positions[ 0 ], positions[ 1 ] }, weights, nb_diracs, N<flags>() );
        RDTSC_FINAL( t1_grid );

        CP b( typename CP::Box{ { 0, 0 }, { 1, 1 } } );
        std::vector<std::size_t> nb_cuts( 16 * thread_pool.nb_threads(), 0 );
        std::uint64_t t0_each = 0, t1_each = 0;
        RDTSC_START( t0_each );
        grid.for_each_laguerre_cell( [&]( CP &cp, std::size_t /*num*/, int num_thread ) {
            nb_cuts[ 16 * num_thread ] += cp.nb_nodes();
        }, b, { positions[ 0 ], positions[ 1 ] }, weights, nb_diracs, N<flags>() );
        RDTSC_FINAL( t1_each );

        double dt_grid = double( t1_grid - t0_grid ) / nb_diracs;
        double dt_each = double( t1_each - t0_each ) / nb_diracs;
        double dt_sum  = double( t1_each - t0_grid ) / nb_diracs;
        P( nb_diracs_per_cell, dt_grid, dt_each, dt_sum );
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
    std::map<std::size_t,double> nb_cycle_per_cell;

    test<Pc<2>>( nb_cycle_per_cell, "/data/random_n10_d2.npy", N<true>() );

    std::cout << "    \\addplot coordinates {\n";
    for( auto p : nb_cycle_per_cell )
        std::cout << "       ( " << p.first << ", " << p.second << " )\n";
    std::cout << "    };\n";
}