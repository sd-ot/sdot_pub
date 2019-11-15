#include "../../src/sdot/Grids/LGrid.h"
#include "../../src/sdot/Support/P.h"
using namespace sdot;

// // nsmake cxx_name clang++

// // nsmake cpp_flag -march=native
// // nsmake cpp_flag -ffast-math
// // nsmake cpp_flag -O3

template<class Grid>
void display( Grid &grid, std::string filename, std::string grid_filemane = {} ) {
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
        VtkOutput voc( { "weight", "err" } );
        CP ic( typename CP::Box{ { 0, 0 }, { 1, 1 } }, nullptr );
        grid.for_each_laguerre_cell( [&]( auto &cp, auto &d, int ) {
            m.lock();

            cp.display_vtk( voc, { d.weight, d.r } );
            area += cp.integral();

            m.unlock();
        }, ic );

        voc.save( filename );
        P( area );
    }
}

template<class TF>
struct MtVal {
    MtVal( TF val = 0 ) : values( thread_pool.nb_threads(), val ) {}

    TF &operator[]( int num_thread ) { return values[ num_thread ]; }
    TF sum        () const { TF res = 0; for( TF v : values ) res += v; return res; }

    std::vector<TF> values;
};

template<class Pc>
void test_with_Pc() {
    //    constexpr int flags = 0;
    using         Dirac = typename Pc::Dirac;
    using         Grid  = LGrid<Pc>;
    constexpr int dim   = Pc::dim;
    using         CP    = typename Grid::CP;
    using         Pt    = typename Grid::Pt;
    using         TF    = typename Grid::TF;
    using         TI    = typename Grid::TI;

    // load
    std::size_t nb_diracs = 10000;
    std::vector<Dirac> diracs( nb_diracs );
    for( std::size_t n = 0; n < nb_diracs; ++n ) {
        for( std::size_t d = 0; d < dim; ++d )
            diracs[ n ].pos[ d ] = 0.2 + 0.6 * rand() / RAND_MAX;
        diracs[ n ].weight = 1 * sin( diracs[ n ].pos.x ) * sin( diracs[ n ].pos.y );
        diracs[ n ].index = n;
    }

    // grid
    Grid grid( 20 );
    grid.construct( diracs.data(), diracs.size() );

    //
    display( grid, "vtk/pd.vtk", "vtk/grid.vtk" );

    // solve init => first residual,
    MtVal<TF> h;
    TF target_mass = TF( 1 )/ nb_diracs;
    CP ic( typename CP::Box{ { 0, 0 }, { 1, 1 } }, nullptr );
    grid.for_each_laguerre_cell( [&]( CP &cp, Dirac &d, int num_thread ) {
        d.r = target_mass;
        d.p = 0;
        d.m = 0;
        // d.z = M @ r;

        h[ num_thread ] += d.r * d.z;
    }, ic );

    //

    std::vector<TF> err( sdot::thread_pool.nb_threads(), 0 );

    // init
    //    std::mutex m;
    //    std::vector<int> ns( nb_diracs, 0 );
    //    std::sort( ns.begin(), ns.end() );
    //    P( ns );


    //    for( std::size_t num_iter = 0; num_iter < 100; ++num_iter ) {
    //        //
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
}



int main() {
    struct Pc {
        enum { store_the_normals = false };
        enum { allow_ball_cut    = false };
        enum { w_bounds_order    = 1     };
        enum { dim               = 2     };
        using  TI                = std::uint64_t;
        using  TF                = double;
        using  Pt                = Point2<TF>;

        struct Dirac {
            TF weight;
            TI index;
            Pt pos;

            // conjugate gradient
            TF r = 0; ///<
            TF p = 0; ///<
            TF z = 0; ///<
            TF m = 0; ///< inv diag
            TF q = 0; ///<

        };
    };


    test_with_Pc<Pc>();
}
