#include "../../../src/sdot/Grids/LGrid.h"
#include "../../../src/sdot/Support/P.h"
using namespace sdot;

// // nsmake cxx_name clang++
// // nsmake cpp_flag -march=native

// // nsmake cpp_flag -ffast-math
// // nsmake cpp_flag -O3

template<class Grid>
void display( Grid &grid, std::string filename, std::string grid_filemane = {} ) {
    using Di = typename Grid::Dirac;
    using CP = typename Grid::CP;
    using TF = typename Grid::TF;
    std::mutex m;

    if ( grid_filemane.size() ) {
        VtkOutput vog;
        grid.display_vtk( vog, { .weight_elevation = 1 } );
        vog.save( grid_filemane );
    }

    if ( filename.size() ) {
        TF area = 0;
        VtkOutput voc( Di::names() );
        typename CP::Box box{ { TF( 0 ) }, { TF( 1 ) } };
        grid.for_each_laguerre_cell( [&]( auto &cp, auto &d, int ) {
            m.lock();

            cp.display_vtk( voc, d.values() );
            area += cp.integral();

            m.unlock();
        }, box );

        voc.save( filename );
        P( area );
    }
}

template<class TF>
struct MtVal {
    enum {          sep       = 16 };

    /**/            MtVal     ( TF val = 0 ) : values( sep * thread_pool.nb_threads(), val ) {}
    void            clear     () { for( std::size_t i = 0; i < values.size(); i += sep ) values[ i ] = 0; }

    TF             &operator[]( int num_thread ) { return values[ sep * num_thread ]; }
    TF              sum       () const { TF res = 0; for( std::size_t i = 0; i < values.size(); i += sep ) res += values[ i ]; return res; }

    std::vector<TF> values;
};

template<class Pc>
void test_with_Pc() {
    using Dirac = typename Pc::Dirac;
    using Grid  = LGrid<Pc>;
    using CP    = typename Grid::CP;
    using Pt    = typename Grid::Pt;
    using TF    = typename Grid::TF;
    using TI    = typename Grid::TI;

    // load
    std::size_t nb_diracs = 20;
    std::vector<Dirac> diracs( nb_diracs );
    for( std::size_t n = 0; n < nb_diracs; ++n ) {
        diracs[ n ].pos[ 0 ] = 1.0 * rand() / RAND_MAX;
        diracs[ n ].pos[ 1 ] = 1.0 * rand() / RAND_MAX;
        diracs[ n ].pos[ 2 ] = 1.0 * rand() / RAND_MAX;
        diracs[ n ].index = n;
    }

    // grid
    Grid grid( 20 );
    grid.construct( diracs.data(), diracs.size() );

    // solve init => first residual,
    display( grid, "vtk/pd.vtk", "vtk/grid.vtk" );
}



int main() {
    struct Pc {
        enum { store_the_normals = false };
        enum { allow_ball_cut    = false };
        enum { w_bounds_order    = 0     };
        enum { dim               = 3     };
        using  TI                = std::uint64_t;
        using  TF                = double;
        using  Pt                = Point3<TF>;

        struct Dirac {
            static std::vector<std::string> names() { return { "weight", "index" }; }
            std::vector<TF> values() const { return { weight, TF( index ) }; }

            TF weight;
            TI index;
            Pt pos;
        };
    };

    test_with_Pc<Pc>();
}
