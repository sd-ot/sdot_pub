#include "../../src/sdot/Support/StaticRange.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
#include "../../src/sdot/ZGrid.h"
#include <fstream>
#include <map>
using namespace sdot;

// // nsmake cpp_flag -march=skylake
// // nsmake cxx_name clang++

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

int main() {
    using Grid = ZGrid<Pc>;
    using Pt = Grid::Pt;
    using TF = Grid::TF;

    std::vector<Pt> positions;
    std::vector<TF> weights;
    for( std::size_t i = 0; i < 1000; ++i ) {
        TF x = double( rand() ) / RAND_MAX;
        TF y = double( rand() ) / RAND_MAX;
        positions.push_back( { 0.0 + 0.05 * x + 0.10 * y, y } );
        positions.push_back( { 1.0 - 0.05 * x - 0.15 * y, y } );
        weights.push_back( 0.0 );
        weights.push_back( 0.0 );
    }

    Grid grid( 10 );
    grid.update( positions.data(), weights.data(), weights.size() );


    VtkOutput vo;
    grid.display( vo );
    vo.save( "vtk/grid.vtk" );
}
