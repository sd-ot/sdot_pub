#include "../../src/sdot/Support/StaticRange.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
#include "../../src/sdot/Grids/LGrid.h"
#include <cnpy.h>
#include <map>
using namespace sdot;

// // nsmake cxx_name clang++
// // nsmake cpp_flag -ffast-math
// // nsmake cpp_flag -march=skylake

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O3
//// nsmake lib_name z

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

        //
        TF weight;
        TI index;
        Pt pos;
        TF dxn;
    };
};

template<class Pc>
void test( std::map<std::size_t,double> &nb_cycle_per_cell, std::string filename ) {
    constexpr int dim = Pc::dim;
    using Grid = LGrid<Pc>;

    using Dirac = typename Grid::Dirac;
    using CP    = typename Grid::CP;
    using Pt    = typename Grid::Pt;
    using TF    = typename Grid::TF;
    using TI    = typename Grid::TI;

    // load
    cnpy::NpyArray arr = cnpy::npy_load( filename );
    std::size_t nb_diracs = arr.shape[ 1 ];
    double *data = arr.data<double>();

    std::vector<typename Pc::Dirac> diracs( nb_diracs );
    for( std::size_t n = 0; n < nb_diracs; ++n ) {
        for( std::size_t d = 0; d < dim; ++d )
            diracs[ n ].pos[ d ] = data[ n + d * nb_diracs ];
        diracs[ n ].weight = data[ n + dim * nb_diracs ];
    }

    // get timings
    double best_dt_sum = 1e6, smurf = 0;
    for( std::size_t nb_diracs_per_cell = 25; nb_diracs_per_cell <= 25; nb_diracs_per_cell += 1 ) {
        // constexpr int flags = Grid::homogeneous_weights * voronoi;
        // RaiiTime re("total");

        std::uint64_t t0 = 0, t1 = 0, nb_reps = 10;
        RDTSC_START( t0 );
        for( std::size_t rep = 0; rep < nb_reps; ++rep ) {
            Grid grid( nb_diracs_per_cell );
            grid.construct( diracs.data(), nb_diracs );

            std::vector<std::size_t> nb_cuts( 16 * thread_pool.nb_threads(), 0 );
            typename CP::Box box{ TF( 0 ), TF( 1 ) };
            grid.for_each_laguerre_cell( [&]( auto &cp, auto &/*dirac*/, int num_thread ) {
                nb_cuts[ 16 * num_thread ] += cp.nb_nodes();
            }, box );

            smurf += nb_cuts[ 0 ];
        }
        RDTSC_FINAL( t1 );

        double dt = double( t1 - t0 ) / nb_diracs / nb_reps;
        best_dt_sum = std::min( best_dt_sum, dt );
        P( nb_diracs_per_cell, dt );

        //        VtkOutput vo;
        //        grid.display_vtk( vo, { .weight_elevation = 1.0 } );
        //        vo.save( "vtk/grid.vtk" );
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

    const char *filenames[] = {
        "/data/sdot/uniform_12500_3D_solved.npy",
        "/data/sdot/uniform_25000_3D_solved.npy",
        "/data/sdot/uniform_50000_3D_solved.npy",
        "/data/sdot/uniform_100000_3D_solved.npy",
        "/data/sdot/uniform_200000_3D_solved.npy",
    };

    for( const char *filename : filenames )
        test<Pc<3>>( nb_cycle_per_cell, filename );

    //    const char *filenames[] = {
    //        "/data/sdot/uniform_100000_2D_solved.npy",
    //        "/data/sdot/uniform_200000_2D_solved.npy",
    //        "/data/sdot/uniform_400000_2D_solved.npy",
    //        "/data/sdot/uniform_800000_2D_solved.npy",
    //        "/data/sdot/uniform_1600000_2D_solved.npy",
    //    };

    //    for( const char *filename : filenames )
    //        test<Pc<2>>( nb_cycle_per_cell, filename, N<true>() );

    std::cout << "    \\addplot coordinates {\n";
    for( auto p : nb_cycle_per_cell )
        std::cout << "       ( " << p.first << ", " << p.second << " )\n";
    std::cout << "    };\n";
}
