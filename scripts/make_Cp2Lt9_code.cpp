#include "../src/sdot/Support/Display/va_string.h"
#include "Cp2Lt9/Cp2Lt9_Func.h"
#include <iostream>
#include <fstream>
#include <map>

void gen( std::string float_type, std::string simd_type, int min_nb_nodes, int max_nb_nodes ) {
    // best parm set
    Cp2Lt9_Func best_func;
    OptParm opt_parm;
    do {
        Cp2Lt9_Func trial_func( opt_parm, float_type, simd_type, min_nb_nodes, max_nb_nodes );
        if ( best_func.score() > trial_func.score() )
            best_func = trial_func;
    } while ( opt_parm.inc() );

    // write code
    std::string fout_name = va_string( "src/sdot/ConvexPolyhedron/Internal/ConvexPolyhedron2dLt64_cut_{}_{}.h", float_type, simd_type );
    std::ofstream fout( fout_name.c_str() );
    best_func.write_def( fout );
}

int main() {
    int min_nb_nodes = 3;
    int max_nb_nodes = 9;

    gen( "gen", "gen", min_nb_nodes, max_nb_nodes );

    //    for( std::string float_type : { "double", "float" } )
    //        for( std::string simd_type : { "SSE2", "AVX2", "AVX512" } )
    //            write_for( float_type, simd_type, max_nb_nodes );
}
