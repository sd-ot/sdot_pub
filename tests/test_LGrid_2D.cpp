#include "../../src/sdot/Support/P.h"
#include "../../src/sdot/LGrid.h"
using namespace sdot;

// // nsmake cxx_name clang++
// // nsmake cpp_flag -ffast-math
// // nsmake cpp_flag -march=native
// // nsmake cpp_flag -O2

template<int _dim>
struct Pc {
    enum { store_the_normals = false };
    enum { allow_ball_cut    = false };
    enum { dim               = _dim };
    using  TF                = double;
    using  TI                = std::size_t;
    using  CI                = std::size_t;
};

template<class Pc>
void test_with_Pc() {
    using         Grid = LGrid<Pc>;
    constexpr int dim  = Pc::dim;
    using         CP   = typename Grid::CP;
    using         Pt   = typename Grid::Pt;
    using         TF   = typename Grid::TF;
    using         TI   = typename Grid::TI;

    // load
    std::size_t nb_diracs = 100;

    std::vector<>
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
}

int main() {
    test_with_Pc<Pc<2>>();
}
