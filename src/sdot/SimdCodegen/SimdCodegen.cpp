#include "../Support/TODO.h"
#include "SimdCodegen.h"

SimdCodegen::SimdCodegen( int simd_size ) : simd_size( simd_size ) {
}

Path SimdCodegen::best_path_for( std::vector<int> res ) {
    // wanted values
    std::vector<std::size_t> wan;
    for( std::size_t i = 0; i < res.size(); ++i )
        if ( res[ i ] >= 0 )
            wan.push_back( i );

    // two consecutive values ?
    if ( wan.size() == 2 && wan[ 0 ] + 1 == wan[ 1 ] && wan[ 0 ] % simd_size == 0 ) {
        int rem_reg_0 = res[ wan[ 0 ] ] % simd_size, rem_reg_1 = res[ wan[ 1 ] ] % simd_size;
        int num_reg_0 = res[ wan[ 0 ] ] / simd_size, num_reg_1 = res[ wan[ 0 ] ] / simd_size;
        int out_reg = wan[ 0 ] / simd_size;

        // in the same register ?
        if ( num_reg_0 == num_reg_1 ) {
            if ( rem_reg_0 == 0 && rem_reg_1 == 1 )
                return { { new Instruction256Cast128( out_reg, num_reg_0 ) } };
            if ( rem_reg_0 == 2 && rem_reg_1 == 3 )
                return { { new Instruction256Extract128( out_reg, num_reg_0, 1 ) } };

            if ( rem_reg_0 == 0 && rem_reg_1 == 2 )
                return { { new Instruction256Perm( out_reg, num_reg_0, { 0, 2, 0, 0 } ) } };
        }
    }

    TODO;
}
