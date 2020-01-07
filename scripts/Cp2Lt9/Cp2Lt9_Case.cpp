#include "Cp2Lt9_Case.h"
#include <sstream>
#include <limits>

Cp2Lt9_Case::Cp2Lt9_Case( OptParm &opt_parm, int nb_nodes, unsigned comb, int simd_size ) : Cp2Lt9_Case() {
    this->simd_size = simd_size;

    for( int i = 0; i < nb_nodes; ++i )
        outside.push_back( comb & ( 1 << i ) );

    cut_list = { outside };

    std::ostringstream ss;
    make_code( ss );
    code = ss.str();
}

Cp2Lt9_Case::Cp2Lt9_Case() : sp( "            " ), valid( true ) {
}

void Cp2Lt9_Case::make_code( std::ostream &os ) {
    // fully inside
    if ( cut_list.split_indices().size() == 0 ) {
        os << sp << "// fully inside (should not happen at this point)\n";
        os << sp << "continue;\n";
        return;
    }

    // fully outside
    if ( cut_list.split_indices().size() == 0 ) {
        os << sp << "// fully outside\n";
        if ( outside.size() )
            os << sp << "nodes_size = 0;\n";
        os << sp << "return true;\n";
        return;
    }

    // uncommon cases
    if ( cut_list.split_indices().size() != 2 ) {
        valid = false;
        return;
    }

    os << "            // " << cut_list << "\n";
}
