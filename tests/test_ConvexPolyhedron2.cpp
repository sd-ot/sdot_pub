#include "../src/sdot/Geometry/ConvexPolyhedron2.h"
#include "../src/sdot/Support/ASSERT.h"
#include "../src/sdot/Support/P.h"
#include "../src/sdot/VtkOutput.h"
using namespace sdot;
//// nsmake cpp_flag -march=native

int main() {
    struct Pc {
        using TF = double;
        using TI = std::size_t;
        using CI = std::size_t;

    };
    using Cp = ConvexPolyhedron2<Pc>;

    Cp cp( Cp::Box{ { 0, 0 }, { 1, 1 } } );
    cp.plane_cut( { 0.5, 0.5 }, { 1.0, 1.0 }, 17, N<0>() );

    VtkOutput vo( { "smurf" } );
    cp.display( vo, { 1.0 } );
    vo.save( "vtk/pd.vtk" );
}
