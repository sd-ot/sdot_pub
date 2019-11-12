#include "../src/sdot/Grids/LGrid.h"
#include "../src/sdot/Support/P.h"
using namespace sdot;

// // nsmake cxx_name clang++
// // nsmake cpp_flag -ffast-math
// // nsmake cpp_flag -march=native
// // nsmake cpp_flag -O2

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
    std::size_t nb_diracs = 1000;
    std::vector<Pt> positions( nb_diracs );
    std::vector<TF> weights( nb_diracs );
    for( std::size_t n = 0; n < nb_diracs; ++n ) {
        for( std::size_t d = 0; d < dim; ++d )
            positions[ n ][ d ] = 0.1 + 0.8 * rand() / RAND_MAX;
        weights[ n ] = 1e-1 * sin( positions[ n ].x ) * sin( positions[ n ].y );
    }

    // get timings
    Grid grid( 20 );
    grid.update( positions.data(), weights.data(), nb_diracs, N<0>() );
    // PN( grid );

    VtkOutput vog;
    grid.display( vog, 1 );
    vog.save( "vtk/grid.vtk" );

    TF area = 0;
    std::mutex m;
    VtkOutput voc;
    CP ic( typename CP::Box{ { 0, 0 }, { 1, 1 } } );
    grid.for_each_laguerre_cell( [&]( CP &cp, std::size_t /*num*/, int /*num_thread*/ ) {
        m.lock();
        cp.display( voc );
        area += cp.integral();
        m.unlock();
    }, ic, N<0>() );
    voc.save( "vtk/pd.vtk" );

    P( area );
}

int main() {
    test_with_Pc<Pc<2>>();
}
