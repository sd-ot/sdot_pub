#include "Cp2Lt9_Case.h"
#include <sstream>
#include <limits>

Cp2Lt9_Case::Cp2Lt9_Case( OptParm &opt_parm, int nb_nodes, unsigned comb, int simd_size ) : Cp2Lt9_Case() {
    for( int i = 0; i < nb_nodes; ++i )
        outside.push_back( comb & ( 1 << i ) );

    std::ostringstream ss:
    make_code( ss );
    code = ss.str();
}

Cp2Lt9_Case::Cp2Lt9_Case() {
}

void Cp2Lt9_Case::make_code( std::ostream &os ) {

}
