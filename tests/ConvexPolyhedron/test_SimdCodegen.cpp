#include "../../src/sdot/SimdCodegen/SimdGraph.h"
#include "../../src/sdot/Support/P.h"
#include <iostream>

int main( int /*argc*/, char **/*argv*/ ) {
    SimdGraph gr;

    gr.add_target(
        gr.make_op( "+", {
            gr.make_op( "R0", {} ),
            gr.make_op( "R1", {} ),
        } )
    );

    gr.display();
}
