#include "../../src/sdot/Support/StaticRange.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
#include "../../src/sdot/Grids/LGrid.h"
#include <map>
using namespace sdot;

// // nsmake cxx_name clang++
// // nsmake cpp_flag -ffast-math
// // nsmake cpp_flag -march=skylake

// // nsmake cpp_flag -march=native
// // nsmake cpp_flag -O2

template<int _dim>
struct Pc {
    enum { store_the_normals = false };
    enum { allow_ball_cut    = false };
    enum { w_bounds_order    = 1     };
    enum { dim               = _dim  };
    using  TF                = double;
    using  TI                = std::size_t;
    using  CI                = std::size_t;
    using  Pt                = typename PointTraits<TF,dim>::type;

    struct Dirac {
        static std::vector<std::string> names() { return { "weight", "index", "dxn" }; }
        std::vector<TF> values() const { return { weight, TF( index ), dxn }; }
        void write_to_stream( std::ostream &os ) const { os << pos; } //index; }

        //
        TF weight;
        TI index;
        Pt pos;
        TF dxn;
    };
};

template<class Dirac>
void make_divs( std::vector<Dirac> &diracs, int n, Point2<double> mi, Point2<double> dl ) {
    if ( n ) {
        #define PD( U, V ) make_divs( diracs, n - 1, { mi[ 0 ] + U * dl[ 0 ], mi[ 1 ] + V * dl[ 1 ] }, 0.5 * dl )
        PD( 0.0, 0.0 );
        PD( 0.5, 0.0 );
        PD( 0.0, 0.5 );
        PD( 0.5, 0.5 );
        return;
    }
    diracs.push_back( { 0.0, diracs.size(), mi, 0.0 } );
}

template<class Pc>
void test() {
    constexpr int dim = Pc::dim;
    using Grid = LGrid<Pc>;

    using Dirac = typename Grid::Dirac;
    using CP    = typename Grid::CP;
    using Pt    = typename Grid::Pt;
    using TF    = typename Grid::TF;
    using TI    = typename Grid::TI;

    using std::min;

    // load
    Grid grid( 5 );
    TI nb_diracs = 50;
    grid.ooc_dir = "ooc/";
    grid.max_usable_ram = 2000;
    grid.nb_fcells_per_ooc_file = 2;
    grid.construct( [&]( const std::function<void( const Dirac *diracs, TI nb_diracs, bool ptrs_survive_the_call )> &cb ) {
        srand( 0 );
        std::vector<Dirac> loc_diracs( 1e3 );
        for( TI n = 0; n < nb_diracs; ) {
            TI r = min( nb_diracs, n + loc_diracs.size() ) - n;
            for( TI o = 0; o < r; ++o, ++n ) {
                Pt p;
                for( std::size_t d = 0; d < dim; ++d )
                    p[ d ] = 1.0 * rand() / RAND_MAX;
                loc_diracs[ o ] = Dirac{ 0, n, p, 0.0 };
            }

            cb( loc_diracs.data(), r, false );
        }

        // std::vector<Dirac> loc_diracs;
        // make_divs( loc_diracs, 3, { 0.0, 0.0 }, { 1.0, 1.0 } );
        // cb( loc_diracs.data(), loc_diracs.size(), false );
    } );

    //PN( grid );

    VtkOutput vo;
    grid.display_vtk( vo, { .weight_elevation = 1.0, .display_cells = true, .display_boxes = false } );
    vo.save( "vtk/grid.vtk" );


    //    // get timings
    //    double best_dt_sum = 1e6, smurf = 0;
    //    for( std::size_t nb_diracs_per_cell = 23; nb_diracs_per_cell <= 43; nb_diracs_per_cell += 1 ) {
    //        // constexpr int flags = Grid::homogeneous_weights * voronoi;
    //        // RaiiTime re("total");

    //        std::uint64_t t0 = 0, t1 = 0, nb_reps = 1;
    //        RDTSC_START( t0 );
    //        for( std::size_t rep = 0; rep < nb_reps; ++rep ) {
    //            Grid grid( nb_diracs_per_cell );
    //            grid.construct( diracs.data(), nb_diracs );

    //            std::vector<std::size_t> nb_cuts( 16 * thread_pool.nb_threads(), 0 );
    //            typename CP::Box box{ { 0, 0 }, { 1, 1 } };
    //            grid.for_each_laguerre_cell( [&]( auto &cp, auto &/*dirac*/, int num_thread ) {
    //                nb_cuts[ 16 * num_thread ] += cp.nb_nodes();
    //            }, box );

    //            smurf += nb_cuts[ 0 ];
    //        }
    //        RDTSC_FINAL( t1 );

    //        double dt = double( t1 - t0 ) / nb_diracs / nb_reps;
    //        best_dt_sum = std::min( best_dt_sum, dt );
    //        P( nb_diracs_per_cell, dt );

    //        //        VtkOutput vo;
    //        //        grid.display_vtk( vo, { .weight_elevation = 1.0 } );
    //        //        vo.save( "vtk/grid.vtk" );
    //    }
    //    P( nb_diracs, smurf, best_dt_sum );
}

int main() {
    test<Pc<2>>();
}
