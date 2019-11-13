#include "../src/sdot/Grids/LGrid.h"
#include "../src/sdot/Support/P.h"
using namespace sdot;

// // nsmake cxx_name clang++

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O3

template<int _dim>
struct Pc {
    enum { store_the_normals = false };
    enum { allow_ball_cut    = false };
    enum { w_bounds_order    = 1     };
    enum { dim               = _dim  };
    using  TI                = std::uint64_t;
    using  SI                = std::int64_t;
    using  CI                = std::size_t;
    using  TF                = double;

    struct Af                { TF new_weight; };
};

template<class Pc>
void test_with_Pc() {
    constexpr int flags = 0;
    using         Grid  = LGrid<Pc>;
    constexpr int dim   = Pc::dim;
    using         CP    = typename Grid::CP;
    using         Pt    = typename Grid::Pt;
    using         TF    = typename Grid::TF;
    using         TI    = typename Grid::TI;

    // load
    std::size_t nb_diracs = 1000;
    std::vector<Pt> positions( nb_diracs );
    std::vector<TF> weights( nb_diracs );
    for( std::size_t n = 0; n < nb_diracs; ++n ) {
        for( std::size_t d = 0; d < dim; ++d )
            positions[ n ][ d ] = 0.1 + 0.8 * rand() / RAND_MAX;
        weights[ n ] = 1e-1 * sin( positions[ n ].x ) * sin( positions[ n ].y );
    }

    // grid
    Grid grid( 20 );
    grid.update_positions_and_weights( positions.data(), weights.data(), nb_diracs, N<flags>() );

    VtkOutput vog;
    grid.display( vog, 1 );
    vog.save( "vtk/grid.vtk" );

    // solve
    TF target_mass = TF( 1 )/ nb_diracs;
    CP ic( typename CP::Box{ { 0, 0 }, { 1, 1 } } );
    for( std::size_t num_iter = 0; num_iter < 100; ++num_iter ) {
        std::vector<TF> err( sdot::thread_pool.nb_threads(), 0 );

        grid.for_each_laguerre_cell( [&]( CP &cp, int num_thread ) {
            TF e = cp.integral() - target_mass;
            err[ num_thread ] += e * e;

            cp.af->new_weight = *cp->dirac_weight - 1e-1 * e;
        }, ic, N<flags>() );

        for( std::size_t i = 1; i < err.size(); ++i )
            err[ 0 ] += err[ i ];
        P( err[ 0 ] );

        grid.assign_new_weights( N<flags>() );
    }

    // display
    std::vector<VtkOutput> voc( sdot::thread_pool.nb_threads(), { { "weight", "dw" } } );
    grid.for_each_laguerre_cell( [&]( CP &cp, int num_thread ) {
        cp.display( voc[ num_thread ], { *cp.dirac_weight, 17 } );
    }, ic, N<flags>() );
    for( std::size_t i = 1; i < voc.size(); ++i )
        voc[ 0 ].append( voc[ i ] );
    voc[ 0 ].save( "vtk/pd.vtk" );
}

int main() {
    test_with_Pc<Pc<2>>();
}
