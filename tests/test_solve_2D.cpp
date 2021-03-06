#include "../src/sdot/Domains/ConvexPolyhedronAssembly.h"
#include "../src/sdot/Grids/LGrid.h"
#include "../src/sdot/display_vtk.h"
#include "../src/sdot/Support/P.h"
using namespace sdot;

// // nsmake cxx_name clang++

// // nsmake cpp_flag -march=native
// // nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O3

//template<class Grid,int flags>
//void display( Grid &grid, std::string filename, N<flags> ) {
//    using CP = typename Grid::CP;
//    using TF = typename Grid::TF;
//    std::mutex m;

//    TF area = 0;
//    VtkOutput voc( { "weight", "err" } );
//    CP ic( typename CP::Box{ { 0, 0 }, { 1, 1 } } );
//    grid.for_each_laguerre_cell( [&]( auto &cp, int ) {
//        m.lock();
//        cp.display( voc, { *cp.dirac_weight, cp.dirac_af->err } );
//        area += cp.integral();
//        m.unlock();
//    }, ic, N<flags>() );

//    P( area );
//    voc.save( filename );
//}

template<class Pc>
void test_with_Pc() {
    using         Domain = ConvexPolyhedronAssembly<Pc>;
    using         Grid   = LGrid<Pc>;
    using         CP     = typename Grid::CP;
    using         Pt     = typename Grid::Pt;
    using         TF     = typename Grid::TF;
    using         TI     = typename Grid::TI;

    constexpr int flags  = 0;
    constexpr int dim    = Pc::dim;

    // load
    std::size_t nb_diracs = 10000;
    std::vector<TF> weights( nb_diracs );
    std::vector<Pt> positions( nb_diracs );
    for( std::size_t n = 0; n < nb_diracs; ++n ) {
        for( std::size_t d = 0; d < dim; ++d )
            positions[ n ][ d ] = 0.2 + 0.6 * rand() / RAND_MAX;
        weights[ n ] = 0 * sin( positions[ n ].x ) * sin( positions[ n ].y );
    }

    // grid
    Grid grid( 20 );
    grid.update_positions_and_weights( positions.data(), weights.data(), nb_diracs, N<flags>() );

    // domain
    Domain domain;
    domain.add_box( { 0, 0 }, { 1, 1 } );

    //
    display_vtk( domain, grid, "vtk/pd.vtk" );

    //    // solve
    //    TF target_mass = TF( 1 )/ nb_diracs;
    //    CP ic( typename CP::Box{ { 0, 0 }, { 1, 1 } } );
    //    for( std::size_t num_iter = 0; num_iter < 100; ++num_iter ) {
    //        std::vector<TF> err( sdot::thread_pool.nb_threads(), 0 );

    //        grid.for_each_laguerre_cell( [&]( CP &cp, int num_thread ) {
    //            TF e = cp.integral() - target_mass;
    //            err[ num_thread ] += e * e;
    //            cp.dirac_af->err = e;
    //        }, ic, N<flags>() );

    //        for( std::size_t i = 1; i < err.size(); ++i )
    //            err[ 0 ] += err[ i ];
    //        P( err[ 0 ] );

    //        grid.mod_weights( [&]( const Pt &/*position*/, TF &weight, typename Pc::Af &af ) {
    //            weight -= 1e-2 * af.err;
    //        } );
    //    }

    //    // display
    //    display( grid, "vtk/pd.vtk", N<flags>() );

    //    VtkOutput vog;
    //    grid.display( vog, 1 );
    //    vog.save( "vtk/grid.vtk" );
}

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
    struct Af                { TF err; };
};

int main() {

    test_with_Pc<Pc<2>>();
}
