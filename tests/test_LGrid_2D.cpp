#include "../src/sdot/Support/P.h"
#include "../src/sdot/LGrid.h"
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
    std::size_t nb_diracs = 10;

    std::vector<TF> data( nb_diracs * ( dim + 1 ) );
    for( std::size_t n = 0; n < nb_diracs; ++n ) {
        for( std::size_t i = 0; i < dim; ++i )
            data[ i * nb_diracs + n ] = 1.0 * rand() / RAND_MAX;
        // data[ dim * nb_diracs + n ] = 0.25 * cos( 3 * data[ 0 * nb_diracs + n ] + 0 * data[ 1 * nb_diracs + n ] );
        data[ dim * nb_diracs + n ] = 0.2 * data[ 0 * nb_diracs + n ];
    }

    std::array<const double *,dim> positions;
    for( std::size_t i = 0; i < dim; ++i )
        positions[ i ] = data.data() + i * nb_diracs;
    double *weights = data.data() + dim * nb_diracs;

    // get timings
    Grid grid( 1 );
    grid.update( positions, weights, nb_diracs, N<0>() );

    PN( grid );

    VtkOutput vo;
    grid.display( vo, 1 );
    vo.save( "vtk/grid.vtk" );

    //    CP b( typename CP::Box{ { 0, 0 }, { 1, 1 } } );
    //    grid.for_each_laguerre_cell( [&]( CP &cp, std::size_t /*num*/, int num_thread ) {
    //    }, b, { positions[ 0 ], positions[ 1 ] }, weights, nb_diracs, N<0>() );
}

int main() {
    test_with_Pc<Pc<2>>();
}
