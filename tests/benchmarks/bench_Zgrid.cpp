#include "../../src/sdot/Support/StaticRange.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
#include "../../src/sdot/ZGrid.h"
#include <fstream>
#include <map>
using namespace sdot;

// // nsmake cxx_name clang++
// // nsmake cpp_flag -ffast-math
// // nsmake cpp_flag -march=skylake

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O2

struct Pc {
    enum { store_the_normals = false };
    enum { allow_ball_cut    = false };
    enum { dim               = 2 };
    using  TF                = double;
    using  TI                = std::size_t;
    using  CI                = std::size_t;

};

template<class Pt,class TF>
void test_vol( const std::vector<TF> &positions_x, const std::vector<TF> &positions_y, const std::vector<TF> &weights ) {
    using Grid = ZGrid<Pc>;
    using CP = Grid::CP;

    Grid grid( 10 );
    grid.update( { positions_x.data(), positions_y.data() }, weights.data(), weights.size(), N<Grid::homogeneous_weights>() );

    CP b( CP::Box{ { 0, 0 }, { 1, 1 } } );
    std::vector<TF> vols( thread_pool.nb_threads(), 0 );
    grid.for_each_laguerre_cell( [&]( CP &cp, std::size_t /*num*/, int num_thread ) {
        vols[ num_thread ] += cp.integral();
    }, b, { positions_x.data(), positions_y.data() }, weights.data(), weights.size(), N<Grid::homogeneous_weights>() );

    TF vol = 0;
    for( TF v : vols )
        vol += v;
    P( vol );
}

int main() {
    using Grid = ZGrid<Pc>;
    using CP = Grid::CP;
    using Pt = Grid::Pt;
    using TF = Grid::TF;

    // thread_pool.init( 1 );

    std::vector<TF> positions_x;
    std::vector<TF> positions_y;
    std::vector<TF> weights;
    for( std::size_t i = 0; i < 10000000; ++i ) {
        TF x = double( rand() ) / RAND_MAX;
        TF y = double( rand() ) / RAND_MAX;
        positions_x.push_back( x );
        positions_y.push_back( y );
        weights.push_back( 0.0 );
        //        TF x = double( rand() ) / RAND_MAX;
        //        TF y = double( rand() ) / RAND_MAX;
        //        positions.push_back( { 0.0 + 0.05 * x + 0.10 * y, y } );
        //        positions.push_back( { 1.0 - 0.05 * x - 0.35 * y, y } );
        //        weights.push_back( 0.0 );
        //        weights.push_back( 0.0 );
    }

    // check if computation is correct
    test_vol( positions_x, positions_y, weights );

    // get timings
    double best_dt_sum = 1e6, smurf = 0;
    for( std::size_t nb_diracs_per_cell = 20; nb_diracs_per_cell < 26; nb_diracs_per_cell += 1 ) {
        std::uint64_t t0_grid = 0, t1_grid = 0;
        RDTSC_START( t0_grid );
        Grid grid( nb_diracs_per_cell );
        grid.update( positions.data(), weights.data(), weights.size(), N<Grid::homogeneous_weights>() );
        RDTSC_FINAL( t1_grid );

        CP b( CP::Box{ { 0, 0 }, { 1, 1 } } );
        std::vector<std::size_t> nb_cuts( thread_pool.nb_threads(), 0 );
        std::uint64_t t0_each = 0, t1_each = 0;
        RDTSC_START( t0_each );
        grid.for_each_laguerre_cell( [&]( CP &cp, std::size_t /*num*/, int num_thread ) {
            nb_cuts[ num_thread ] += cp.nb_nodes();
        }, b, positions.data(), weights.data(), weights.size(), N<Grid::homogeneous_weights>() );
        RDTSC_FINAL( t1_each );

        double dt_grid = double( t1_grid - t0_grid ) / weights.size();
        double dt_each = double( t1_each - t0_each ) / weights.size();
        double dt_sum  = double( t1_each - t0_grid ) / weights.size();
        P( nb_diracs_per_cell, dt_grid, dt_each, dt_sum );
        best_dt_sum = std::min( best_dt_sum, dt_sum );
        smurf += nb_cuts[ 0 ];
    }
    P( smurf, best_dt_sum );

    //    grid.display_tikz( std::cout, 10.0 );
    //    for( std::size_t i = 0; i < positions.size(); ++i )
    //        std::cout << "\\draw[blue] (" << 10 * positions[ i ].x << "," << 10 * positions[ i ].y << ") node {$\\times$};\n";

    //    VtkOutput vo;
    //    grid.display( vo );
    //    vo.save( "vtk/grid.vtk" );
}
