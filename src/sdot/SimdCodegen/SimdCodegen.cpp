#include "../Support/TODO.h"
#include "SimdCodegen.h"

SimdCodegen::SimdCodegen( int simd_size ) : simd_size( simd_size ) {
}

Path SimdCodegen::best_path_for( Reg out, std::vector<Lane> inp ) {
    // two consecutive values ?
    if ( inp.size() == 2 ) {
        int r0 = inp[ 0 ].lane % simd_size, r1 = inp[ 1 ].lane % simd_size;

        // in the same register ?
        if ( inp[ 0 ].reg == inp[ 1 ].reg ) {
            if ( r0 == 0 && r1 == 1 )
                return { { new Instruction256DupEven( out, inp[ 0 ].reg ) } };
            if ( r0 == 0 && r1 == 1 )
                return { { new Instruction256Cast128( out, inp[ 0 ].reg ) } };
            if ( r0 == 0 && r1 == 2 )
                return { { new Instruction256Perm( { 4, "tmp" }, inp[ 0 ].reg, { 0, 2, 2, 3 } ), new Instruction256Cast128( out, { 4, "tmp" } ) } };
            if ( r0 == 0 && r1 == 3 )
                return { { new Instruction256Perm( { 4, "tmp" }, inp[ 0 ].reg, { 0, 3, 2, 3 } ), new Instruction256Cast128( out, { 4, "tmp" } ) } };
            if ( r0 == 2 && r1 == 3 )
                return { { new Instruction256Extract128( out, inp[ 0 ].reg, 1 ) } };

        }
    }

    TODO;
}
