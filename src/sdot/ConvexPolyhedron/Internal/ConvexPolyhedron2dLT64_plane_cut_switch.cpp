#include "../../Support/Display/generic_ostream_output.h"
#include "../../Support/bit_handling.h"
#include "../../Support/ASSERT.h"
#include "../../Support/TODO.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <map>
#include <set>

struct Op {
    /**/        Op             ( std::size_t i0 = 0, std::size_t i1 = 0, int dir = 0 ) : dir( dir ), i0( i0 ), i1( i1 ), sw( false ) {}

    void        write_to_stream( std::ostream &os ) const;
    bool        going_outside  () const { return dir  < 0; }
    bool        going_inside   () const { return dir  > 0; }
    bool        single         () const { return dir == 0; }
    bool        split          () const { return dir != 0; }

    std::size_t outside_node   () const { return dir > 0 ? i0 : i1; }
    std::size_t inside_node    () const { return dir > 0 ? i1 : i0; }

    int         dir;           ///< -1 => going outside. 0 => single node. +1 => going inside.
    std::size_t i0;            ///<
    std::size_t i1;            ///<
    bool        sw;            ///<
};

void Op::write_to_stream( std::ostream &os ) const {
    if ( single() )
        os << inside_node();
    else
        os << "[" << i0 << "," << i1 << "]";
}

struct Mod {
    std::vector<std::size_t> split_indices  () { std::vector<std::size_t> res; for( std::size_t i = 0; i < ops.size(); ++i ) if ( ops[ i ].split() ) res.push_back( i ); return res; }
    void                     rotate         ( std::size_t off ) { std::vector<Op> nops( ops.size() ); for( std::size_t i = 0; i < ops.size(); ++i ) nops[ i ] = ops[ ( i + off ) % ops.size() ]; ops = nops; }
    double                   score          ();
    void                     sw             ( std::uint64_t val ) { std::vector<std::size_t> si = split_indices(); for( std::size_t i = 0; i < si.size(); ++i ) ops[ si[ i ] ].sw = val & ( std::uint64_t( 1 ) << i ); }

    std::vector<Op>          ops;
};


double Mod::score() {
    double res = 0;
    for( std::size_t i = 0; i < ops.size(); ++i ) {
        if ( ops[ i ].single() && ops[ i ].inside_node() != i )
            res += 1.0;
    }
    return res;
}

bool get_code( std::ostringstream &os, std::size_t nb_nodes, std::bitset<8> outside, int simd_size, int nb_regs ) {
    // make a ref Mod
    Mod ref_mod;
    for( std::size_t i = 0; i < nb_nodes; ++i ) {
        if ( outside[ i ] )
            continue;

        // going inside
        std::size_t h = ( i + nb_nodes - 1 ) % nb_nodes;
        if ( outside[ h ] )
            ref_mod.ops.push_back( { h, i, 1 } );

        // inside point
        ref_mod.ops.push_back( { i, i, 0 } );

        // outside point => create points on boundaries
        std::size_t j = ( i + 1 ) % nb_nodes;
        if ( outside[ j ] )
            ref_mod.ops.push_back( { i, j, 1 } );
    }

    // everything is outside
    if ( ref_mod.ops.empty() ) {
        os << "        // everything is outside\n";
        os << "        nodes_size = 0;\n";
        os << "        return fu( *this );\n";
        return true;
    }

    // everything is inside
    if ( ref_mod.split_indices().empty() ) {
        os << "        // everything is inside\n";
        os << "        continue;\n";
        return true;
    }

    // uncommon cases (to reduce code size)
    if ( ref_mod.split_indices().size() > 2 ) {
        return false;
    }

    // find the best permutation
    Mod best_mod;
    double best_score = 1e40;
    for( std::uint64_t sw_val = 0; sw_val < ( 1ul << ref_mod.split_indices().size() ); ++sw_val ) {
        for( std::size_t ro = 0; ro < nb_nodes; ++ro ) {
            Mod mod = ref_mod;
            mod.sw( sw_val );
            mod.rotate( ro );

            double score = mod.score();
            if ( best_score > score ) {
                best_score = score;
                best_mod = mod;
            }
        }
    }

    //
    os << "        // nb_nodes:" << nb_nodes << " outside:" << outside << " ops:" << best_mod.ops << "\n";

    return true;
}

void generate( std::ostream &os, std::string variant ) {
    std::size_t simd_size, nb_regs;
    if ( variant == "AVX2" || variant == "AVX512" ) {
        os << "    using VF = SimdVec<TF,4>;\n";
        os << "    using VC = SimdVec<CI,4>;\n";
        simd_size = 4;
        nb_regs = 2;
    } else {
        TODO;
    }

    // load in registers
    for( std::size_t i = 0; i < nb_regs; ++i ) {
        os << "    VF px_" << i << " = VF::load_aligned( nodes.xs + "      << simd_size * i << " );\n";
        os << "    VF py_" << i << " = VF::load_aligned( nodes.ys + "      << simd_size * i << " );\n";
        os << "    VC pc_" << i << " = VC::load_aligned( nodes.cut_ids + " << simd_size * i << " );\n";
    }
    auto save_regs = [&]( std::ostream &os, std::string sp = "            " ) {
        for( std::size_t i = 0; i < nb_regs; ++i ) {
            os << sp << "VF::store_aligned( nodes.xs + "      << simd_size * i << ", px_" << i << " );\n";
            os << sp << "VF::store_aligned( nodes.ys + "      << simd_size * i << ", py_" << i << " );\n";
            os << sp << "VC::store_aligned( nodes.cut_ids + " << simd_size * i << ", pc_" << i << " );\n";
        }
    };

    // loop
    os << "    for( ; ; ++num_cut ) {\n";
    os << "        if ( num_cut == nb_cuts ) {\n";
    save_regs( os );
    os << "            return fu( *this );\n";
    os << "        }\n";
    os << "    \n";
    os << "        // get distance and outside bit for each node\n";
    os << "        std::uint16_t nmsk = 1 << nodes_size;\n";
    os << "        VF cx = cut_dir[ 0 ][ num_cut ];\n";
    os << "        VF cy = cut_dir[ 1 ][ num_cut ];\n";
    os << "        VC ci = cut_ids[ num_cut ];\n";
    os << "        VF cs = cut_ps[ num_cut ];\n";
    os << "    \n";
    for( std::size_t i = 0; i < nb_regs; ++i )
        os << "        VF bi_" << i << " = px_" << i << " * cx + py_" << i << " * cy;\n";

    os << "        std::uint16_t outside_nodes = (";
    for( std::size_t i = 0; i < nb_regs; ++i )
        os << ( i ? " |" : "" ) << " ( ( bi_" << i << " > cs ) << " << simd_size * i << " )";
    os << " ) & ( nmsk - 1 );\n";

    os << "        std::uint16_t case_code = outside_nodes | nmsk;\n";
    for( std::size_t i = 0; i < nb_regs; ++i )
        os << "        VF di_" << i << " = bi_" << i << " - cs;\n";
    os << "    \n";
    os << "        // if nothing has changed => go to the next cut\n";
    os << "        if ( outside_nodes == 0 )\n";
    os << "            continue;\n";
    os << "    \n";

    // get code for each case
    std::map<std::string,int> code_map; // code content => goto number
    std::vector<int> case_nums; // outside_nodes code => code number
    case_nums.resize( 1 << ( nb_regs * simd_size + 1 ), 0 );
    code_map[ "" ] = 0;

    for( std::size_t nb_nodes = 3; nb_nodes <= nb_regs * simd_size; ++nb_nodes ) {
        for( int outside_case = 0; outside_case < ( 1 << nb_nodes ); ++outside_case ) {
            std::ostringstream code;
            if ( get_code( code, nb_nodes, outside_case, simd_size, nb_regs ) ) {
                // get number
                auto iter = code_map.find( code.str() );
                if ( iter == code_map.end() )
                    iter = code_map.insert( iter, { code.str(), code_map.size() } );

                // store the case
                int code_val = ( 1 << nb_nodes ) | outside_case;
                case_nums[ code_val ] = iter->second;
            }
        }
    }

    // write jump code
    os << "        static void *dispatch_table[] = {";
    for( std::size_t n = 0; n < case_nums.size(); ++n )
        os << ( n % 16 ? " " : "\n            " ) << "&&case_" << case_nums[ n ] << ",";
    os << "\n        };\n";
    os << "        goto *dispatch_table[ case_code ];\n";

    // cases code
    for( auto iter : code_map ) {
        // the generic case is manually written
        if ( iter.second == 0 )
            continue;
        os << "      case_" << iter.second << ": {\n";
        os << iter.first;
        os << "      }\n";
    }
    os << "      case_0: // handled in the next loop\n";
    save_regs( os, "        " );
    os << "        break;\n";

    os << "    }\n";
}

int main( int /*argc*/, char **argv ) {
    // code for IDEs
    std::cout << "#ifdef LOC_PARSE\n";
    std::cout << "#include \"../../Support/SimdVec.h\" \n";
    std::cout << "#include <cstdint>\n";
    std::cout << "using TF = double;\n";
    std::cout << "using CI = std::uint64_t;\n";
    std::cout << "namespace sdot {\n";
    std::cout << "struct Nodes { TF xs[ 64 ], ys[ 64 ]; CI cut_ids[ 64 ]; };\n";
    std::cout << "struct S {\n";
    std::cout << "  void fu( S ) {}\n";
    std::cout << "  void f( Nodes nodes, int &num_cut, int nb_cuts, int nodes_size, TF **cut_dir, CI *cut_ids, TF *cut_ps ) {\n";
    std::cout << "#endif // LOC_PARSE\n";

    generate( std::cout, argv[ 1 ] );

    // code for IDEs
    std::cout << "#ifdef LOC_PARSE\n";
    std::cout << "  }\n";
    std::cout << "};\n";
    std::cout << "} // namespace sdot\n";
    std::cout << "#endif // LOC_PARSE\n";
}

