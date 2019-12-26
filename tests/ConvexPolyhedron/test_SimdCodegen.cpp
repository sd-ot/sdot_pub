#include "../../src/sdot/SimdCodegen/SimdCodegen.h"
#include "../../src/sdot/Support/P.h"
#include <iostream>

//#include <x86intrin.h>

//__m256d f( __m256d px_0, __m256d di_0 ) {
//    for( std::size_t i = 0; i < 100; ++i ) {
//        auto R0 = di_0[ 3 ];
//        auto R1 = di_0[ 2 ];
//        auto R2 = R0 - R1;
//        auto R3 = R1 / R2;
//        auto R4 = di_0[ 1 ];
//        auto R5 = di_0[ 0 ];
//        auto R6 = R4 - R5;
//        auto R7 = R5 / R6;
//        auto R8 = px_0[ 3 ];
//        auto R9 = px_0[ 2 ];
//        auto R10 = R9 - R8;
//        auto R11 = R3 * R10;
//        auto R12 = R9 + R11;
//        auto R13 = px_0[ 2 ];
//        auto R14 = px_0[ 3 ];
//        auto R15 = px_0[ 0 ];
//        auto R16 = px_0[ 1 ];
//        auto R17 = R15 - R16;
//        auto R18 = R7 * R17;
//        auto R19 = R15 + R18;
//        px_0[ 2 ] = R19;
//        px_0[ 1 ] = R14;
//        px_0[ 0 ] = R13;
//        px_0[ 3 ] = R12;
//    }
//    return px_0;
//}


int main( int /*argc*/, char **/*argv*/ ) {
    SimdCodegen sc;

    SimdGraph gr;

    int n0 = 1, n1 = 2, n2 = 2, n3 = 3;
    SimdOp *px_0 = gr.make_op( "REG px_0 d 4", {} );
    SimdOp *di_0 = gr.make_op( "REG di_0 d 4", {} );

    SimdOp *px_a = gr.make_op( "AGG", { gr.get_op( px_0, n0 ), gr.get_op( px_0, n1 ) } );
    SimdOp *px_b = gr.make_op( "AGG", { gr.get_op( px_0, n2 ), gr.get_op( px_0, n3 ) } );
    SimdOp *di_a = gr.make_op( "AGG", { gr.get_op( di_0, n0 ), gr.get_op( di_0, n1 ) } );
    SimdOp *di_b = gr.make_op( "AGG", { gr.get_op( di_0, n2 ), gr.get_op( di_0, n3 ) } );

    SimdOp *di_m = gr.make_op( "DIV", { di_a, gr.make_op( "SUB", { di_b, di_a } ) } );

    SimdOp *adds = gr.make_op( "ADD", { px_a, gr.make_op( "MUL", { di_m, gr.make_op( "SUB", { px_a, px_b } ) } ) } );

    SimdOp *resg = gr.make_op( "AGG", { gr.get_op( px_0, 2 ), gr.get_op( px_0, 3 ), gr.get_op( adds, 0 ), gr.get_op( adds, 1 ) } );

    gr.add_target( gr.make_op( "SET px_0", { resg } ) );
    //    gr.write_code( std::cout, "    " );
    //    gr.display();

    sc.add_possibility( gr );

    sc.write_code( std::cout );
}
