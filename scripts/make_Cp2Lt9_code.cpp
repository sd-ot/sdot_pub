#include "Cp2Lt9/Cp2Lt9_Func.h"
#include <iostream>
#include <fstream>
#include <map>

void gen( std::string float_type, std::string simd_type, int max_nb_nodes ) {
    Cp2Lt9_Func best_func;
    OptParm opt_parm;
    do {
        Cp2Lt9_Func trial_func( opt_parm, float_type, simd_type, max_nb_nodes );
        if ( best_func.score() > trial_func.score() )
            best_func = trial_func;
    } while ( opt_parm.inc() );

    best_func.write_def( std::cout );
}

int main() {
    int max_nb_nodes = 9;

    gen( "gen", "gen", max_nb_nodes );

    //    for( std::string float_type : { "double", "float" } )
    //        for( std::string simd_type : { "SSE2", "AVX2", "AVX512" } )
    //            write_for( float_type, simd_type, max_nb_nodes );
}
