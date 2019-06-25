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
    using CP = Grid::CP;
    using Pt = Grid::Pt;
    using TF = Grid::TF;

    std::vector<Pt> positions;
    std::vector<TF> weights;
    for( std::size_t i = 0; i < 200; ++i ) {
        TF x = double( rand() ) / RAND_MAX;
        TF y = double( rand() ) / RAND_MAX;
        positions.push_back( { x, y } );
        weights.push_back( 0.0 );
        //        TF x = double( rand() ) / RAND_MAX;
        //        TF y = double( rand() ) / RAND_MAX;
        //        positions.push_back( { 0.0 + 0.05 * x + 0.10 * y, y } );
        //        positions.push_back( { 1.0 - 0.05 * x - 0.35 * y, y } );
        //        weights.push_back( 0.0 );
        //        weights.push_back( 0.0 );
    }

    Grid grid( 3 );
    grid.update( positions.data(), weights.data(), weights.size() );

    TF vol = 0;
    std::mutex m;
    CP b( CP::Box{ { 0, 0 }, { 1, 1 } } );
    grid.for_each_laguerre_cell( [&]( CP &cp, std::size_t /*num*/, int /*num_thread*/ ) {
        m.lock();
        vol += cp.integral();
        m.unlock();
    }, b, positions.data(), weights.data(), weights.size() );

    P( vol );

    //    grid.display_tikz( std::cout );
    //    for( std::size_t i = 0; i < positions.size(); ++i )
    //        std::cout << "\\draw[blue] (" << positions[ i ].x << "," << positions[ i ].y << ") node {$\\times$};\n";

    //    VtkOutput vo;
    //    grid.display( vo );
    //    vo.save( "vtk/grid.vtk" );


}
