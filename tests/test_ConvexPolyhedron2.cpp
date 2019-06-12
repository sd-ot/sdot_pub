#include "../src/sdot/Geometry/ConvexPolyhedron2.h"
#include "../src/sdot/Support/ASSERT.h"
#include "../src/sdot/Support/P.h"
#include "../src/sdot/VtkOutput.h"
using namespace sdot;

int main() {
    //    struct Pc {
    //        using TF = double;
    //        using TI = std::size_t;
    //        using CI = std::size_t;

    //    };
    //    using Cp = ConvexPolyhedron2<Pc>;

    //    Cp cp;
    //    cp.resize( 90 );
    //    ASSERT( cp.nb_nodes() == 90, "" );
    //    for( std::size_t i = 0; i < cp.nb_nodes(); ++i ) {
    //        cp.node( i ).x =  1.0 * i;
    //        cp.node( i ).y = 10.0 * i;
    //    }

    //    for( std::size_t i = 0; i < cp.nb_nodes(); ++i ) {
    //        ASSERT( cp.node( i ).x ==  1.0 * i, "" );
    //        ASSERT( cp.node( i ).y == 10.0 * i, "" );
    //    }

    //    int i = 0;
    //    cp.for_each_node( [&]( auto &node ) {
    //        ASSERT( node.x ==  1.0 * i, "" );
    //        ASSERT( node.y == 10.0 * i, "" );
    //        ++i;
    //    } );
    VtkOutput vo;
}
