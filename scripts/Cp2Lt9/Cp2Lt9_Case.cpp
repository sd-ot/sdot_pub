#include "Cp2Lt9_Case.h"
#include <sstream>
#include <limits>

Cp2Lt9_Case::Cp2Lt9_Case( OptParm &opt_parm, int nb_nodes, unsigned comb, int simd_size, int nb_registers ) : Cp2Lt9_Case() {
    this->nb_registers = nb_registers;
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
    // fully outside
    if ( cut_list.ops.size() == 0 ) {
        os << sp << "// fully outside\n";
        if ( outside.size() )
            os << sp << "nodes_size = 0;\n";
        os << sp << "return true;\n";
        return;
    }

    // fully inside
    std::vector<std::size_t> si = cut_list.split_indices();
    if ( si.size() == 0 ) {
        os << sp << "// fully inside (should not happen at this point)\n";
        os << sp << "continue;\n";
        return;
    }

    // uncommon cases
    if ( si.size() != 2 ) {
        valid = false;
        return;
    }

    // helper funcs
    auto val_reg = [&]( std::string c, int n ) {
        if ( n / simd_size < nb_registers )
            return "p" + c + "_" + std::to_string( n / simd_size ) + "[ " + std::to_string( n % simd_size ) + " ]";
        return "p" + c + "[ " + std::to_string( n ) + " ]";
    };

    //
    os << sp << "// out: " << outside << " cut: " << cut_list << "\n";
    if ( cut_list.ops.size() != outside.size() )
        os << sp << "nodes_size = " << cut_list.ops.size() << ";\n";

    for( std::size_t i = 0; i < 2; ++i )
        os << sp << "TF d" << i << " = di_" << cut_list.ops[ si[ i ] ].n0() << " / ( di_" << cut_list.ops[ si[ i ] ].n1() << " - di_" << cut_list.ops[ si[ i ] ].n0() << " );\n";
    for( std::size_t num_op = 0; num_op < cut_list.ops.size(); ++num_op ) {
        const Cp2Lt9_CutList::Cut &op = cut_list.ops[ num_op ];
        if ( op.single() ) {
            if ( num_op == op.inside_node() )
                continue;
            os << sp << val_reg( "x", num_op ) << " = " << val_reg( "x", op.inside_node() ) << ";\n";
        } else {
            os << sp << val_reg( "x", num_op ) << " = " << val_reg( "x", op.n0() ) << " + d0 * ( " << val_reg( "x", op.n0() ) << " - " << val_reg( "x", op.n1() ) << " );\n";
        }
    }

}
