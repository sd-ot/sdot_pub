#include "Cp2Lt9_Case.h"
#include <numeric>
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

    for( std::size_t i = 0; i < 2; ++i ) {
        const Cp2Lt9_CutList::Cut &op = cut_list.ops[ si[ i ] ];
        os << sp << "TF d_" << op.n0() << "_" << op.n1() << " = di_" << op.n0() << " / ( di_" << op.n1() << " - di_" << op.n0() << " );\n";
    }

    //
    std::vector<std::size_t> inds( cut_list.ops.size() );
    std::iota( inds.begin(), inds.end(), 0 );
    std::vector<std::size_t> delayed;
    while ( inds.size() ) {
        // helpers
        auto will_be_needed = [&]( std::size_t num_ind ) {
            for( std::size_t dni_mud = 0; dni_mud < inds.size(); ++dni_mud ) {
                if ( num_ind == dni_mud )
                    continue;
                const Cp2Lt9_CutList::Cut &op = cut_list.ops[ inds[ dni_mud ] ];
                if ( op.n0() == inds[ num_ind ] || op.n1() == inds[ num_ind ] )
                    return true;
            }
            return false;
        };

        auto disp_and_erase = [&]( std::size_t num_ind, int n_tmp = -1 ) {
            std::size_t ind = inds[ num_ind ];
            inds.erase( inds.begin() + num_ind );

            const Cp2Lt9_CutList::Cut &op = cut_list.ops[ ind ];
            if ( op.single() && ind == op.inside_node() )
                return;

            if ( n_tmp >= 0 )
                os << sp << "TF tmpx_" << n_tmp << " = ";
            else
                os << sp << val_reg( "x", ind ) << " = ";

            if ( op.single() ) {
                os << val_reg( "x", op.inside_node() ) << ";\n";
            } else {
                os << val_reg( "x", op.n0() ) << " + d_" << op.n0() << "_" << op.n1() << " * ( "
                   << val_reg( "x", op.n0() ) << " - " << val_reg( "x", op.n1() ) << " );\n";
            }
        };

        // look for an op that does not need a value in inds
        auto direct = [&]() {
            for( std::size_t num_ind = 0; num_ind < inds.size(); ++num_ind ) {
                if ( ! will_be_needed( num_ind ) ) {
                    disp_and_erase( num_ind );
                    return true;
                }
            }
            return false;
        };
        if ( direct() )
            continue;

        //
        delayed.push_back( inds[ 0 ] );
        disp_and_erase( 0, delayed.size() - 1 );
    }

    for( std::size_t i = 0; i < delayed.size(); ++i )
        os << sp << val_reg( "x", delayed[ i ] ) << " = tmpx_" << i << ";\n";
}
