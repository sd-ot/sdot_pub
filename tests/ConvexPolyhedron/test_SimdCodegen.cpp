#include "../../src/sdot/SimdCodegen/SimdGraph.h"
#include "../../src/sdot/Support/P.h"
#include <iostream>

int main( int /*argc*/, char **/*argv*/ ) {
    SimdGraph gr;

    SimdOp *px_0 = gr.make_op( "REG px_0 d 4", {} );
    SimdOp *px_a = gr.make_op( "GET 0", { px_0 } );
    SimdOp *px_b = gr.make_op( "GET 1", { px_0 } );

    SimdOp *di_0 = gr.make_op( "REG di_0 d 4", {} );
    SimdOp *di_a = gr.make_op( "GET 0", { di_0 } );
    SimdOp *di_b = gr.make_op( "GET 1", { di_0 } );

    SimdOp *dm_s = gr.make_op( "DIV", { di_a, gr.make_op( "SUB", { di_b, di_a } ) } );

    SimdOp *aggx = gr.make_op( "AGG", {
        gr.make_op( "GET 2", { px_0 } ),
        gr.make_op( "GET 3", { px_0 } ),
        gr.make_op( "ADD", { px_a, gr.make_op( "MUL", { dm_s, gr.make_op( "SUB", { px_a, px_b } ) } ) } ),
        gr.make_op( "UNK", {} )
    } );

    // SimdVec<TF,2> nx_s = px_a + dm_s * ( px_a - px_b );

    //    SimdOp *di_c = gr.make_op( "EXT 0", { di_0 } );
    //    SimdOp *di_d = gr.make_op( "EXT 1", { di_0 } );

    gr.add_target( aggx );

    gr.display();
}
