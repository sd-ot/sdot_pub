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

/**
  Single operations
*/
struct Op {
    void write_to_stream( std::ostream &os ) const {
        if ( i1 >= 0 )
            os << "[ " << i0 << ", " << i1 << " ]";
        else
            os << i0;
    }

    double cost( int i ) const {
        if ( i1 >= 0 )
            return 1e-2 * ( i0 != i && i1 != i );
        return i0 != i;
    }

    int i0;  ///<
    int i1;  ///< -1 means we take the values only from i0
    int dir; ///< 1 -> going out, -1 -> going in, 0 -> staying inside
};

/**
  Operations
*/
struct Mod {
    void write_to_stream( std::ostream &os ) const {
        os << ops;
    }

    void find_best_rotation() {
        double best_cost = 1e6;
        std::size_t best_offset = 0;
        for( std::size_t i = 0; i < ops.size(); ++i ) {
            double c = cost( i );
            if ( best_cost > c ) {
                best_cost = c;
                best_offset = i;
            }
        }

        std::vector<Op> new_ops;
        for( std::size_t i = 0; i < ops.size(); ++i )
            new_ops.push_back( ops[ ( i + best_offset ) % ops.size() ] );
        ops = std::move( new_ops );
    }

    double cost( std::size_t offset ) const {
        double res = 0;
        for( std::size_t i = 0; i < ops.size(); ++i )
            res += ops[ ( i + offset ) % ops.size() ].cost( i );
        return res;
    }

    //    void write_code_mm256xy( std::ostream &code, int simd_size, int nb_regs ) {
    //        // get op indices for each type of operation
    //        std::vector<std::size_t> permutation_indices, interpolation_indices;
    //        for( std::size_t i = 0; i < std::min( ops.size(), std::size_t( nb_regs * simd_size ) ); ++i ) {
    //            if ( ops[ i ].i1 >= 0 )
    //                interpolation_indices.push_back( i );
    //            else
    //                permutation_indices.push_back( i );
    //        }

    //        // make `idx_0` and `idy_0` (for _mm512_permutex2var_pd)
    //        std::uint64_t idx_0 = 0, idy_0 = 0;
    //        for( std::size_t i = 0; i < interpolation_indices.size(); ++i ) {
    //            idx_0 += ( std::uint64_t( 8 + 2 * i ) << 8 * interpolation_indices[ i ] );
    //            idy_0 += ( std::uint64_t( 9 + 2 * i ) << 8 * interpolation_indices[ i ] );
    //        }
    //        for( std::size_t i : permutation_indices ) {
    //            idx_0 += std::uint64_t( ops[ i ].i0 ) << 8 * i;
    //            idy_0 += std::uint64_t( ops[ i ].i0 ) << 8 * i;
    //        }
    //        code << "    __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << idx_0 << "ul ) );\n";
    //        code << "    __m512i idy_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << idy_0 << "ul ) );\n";
    //        if ( nb_regs > 1 )
    //            TODO;

    //        if ( interpolation_indices.size() == 2 ) {
    //            auto set_pd = [&]( std::string out, std::string x, std::string y, int i0, int i1 ) {
    //                if ( i0 == i1 && x == y )
    //                    code << "    __m256d " << out << " = _mm256_set1_pd( " << x << "_" << i0 << " );\n";
    //                else if ( x == y )
    //                    code << "    __m256d " << out << " = _mm256_set_pd( " << y << "_" << i1 << ", " << x << "_" << i1 << ", " << y << "_" << i0 << ", " << x << "_" << i0 << " );\n";
    //                else
    //                    code << "    __m256d " << out << " = _mm256_set_pd( " << y << "_" << i1 << ", " << x << "_" << i1 << ", " << y << "_" << i0 << ", " << x << "_" << i0 << " );\n";
    //            };

    //            std::size_t it_0 = interpolation_indices[ 0 ];
    //            std::size_t it_1 = interpolation_indices[ 1 ];
    //            const Op &op_0 = ops[ it_0 ];
    //            const Op &op_1 = ops[ it_1 ];

    //            set_pd( "d_i0", "d", "d", op_0.i0, op_1.i0 );
    //            set_pd( "z_i0", "x", "y", op_0.i0, op_1.i0 );

    //            set_pd( "d_i1", "d", "d", op_0.i1, op_1.i1 );
    //            set_pd( "z_i1", "x", "y", op_0.i1, op_1.i1 );

    //            code << "    __m256d m = _mm256_div_pd( d_i0, _mm256_sub_pd( d_i1, d_i0 ) );\n";
    //            code << "    __m512d inter_z = _mm512_castpd256_pd512( _mm256_sub_pd( z_i0, _mm256_mul_pd( m, _mm256_sub_pd( z_i1, z_i0 ) ) ) );\n";
    //        } else {
    //            ASSERT( interpolation_indices.size() == 1, "weird" );
    //            std::size_t it_0 = interpolation_indices[ 0 ];
    //            const Op &op_0 = ops[ it_0 ];

    //            code << "    TF m = d_" << op_0.i0 << " / ( d_" << op_0.i1 << " - d_" << op_0.i0 << " ); // 1\n";
    //            code << "    __m512d inter_z = _mm512_castpd128_pd512( _mm_set_pd( y_" << op_0.i0 << " - m * ( y_" << op_0.i1 << " - y_" << op_0.i0 << " ), x_" << op_0.i0 << " - m * ( x_" << op_0.i1 << " - x_" << op_0.i0 << " ) ) );\n";
    //        }

    //        code << "    px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_z );\n";
    //        code << "    py_0 = _mm512_permutex2var_pd( py_0, idy_0, inter_z );\n";
    //    }

    void write_code_mm128( std::ostream &code, int simd_size, bool inplace = false ) {
        // helper
        auto disp_c = [&]( const Op &op ) {
            if ( op.dir > 0 )
                code << "(long long)cut_id[ num_cut ]";
            else
                code << "(long long)pc[ " << op.i0 << " ]";
        };

        // get op indices for each type of operation
        std::vector<std::size_t> permutation_indices, interpolation_indices;
        for( std::size_t i = 0; i < std::min( ops.size(), std::size_t( simd_size ) ); ++i ) {
            if ( ops[ i ].i1 >= 0 )
                interpolation_indices.push_back( i );
            else
                permutation_indices.push_back( i );
        }

        if ( interpolation_indices.size() == 2 ) {
            if ( inplace ) {
                std::size_t it_0 = interpolation_indices[ 0 ];
                std::size_t it_1 = interpolation_indices[ 1 ];

                int op0_i0 = ops[ it_0 ].i0, op0_i1 = ops[ it_0 ].i1;
                int op1_i0 = ops[ it_1 ].i0, op1_i1 = ops[ it_1 ].i1;
                if ( op0_i0 == op1_i0 )
                    std::swap( op0_i0, op0_i1 );

                // make a `idx_0` (for _mm512_permutex2var_pd)
                std::uint64_t idx_0 = 0;
                idx_0 += std::uint64_t( op0_i0 + 8 ) << 8 * it_0;
                idx_0 += std::uint64_t( op1_i0 + 8 ) << 8 * it_1;
                for( std::size_t i : permutation_indices )
                    idx_0 += std::uint64_t( ops[ i ].i0 ) << 8 * i;
                code << "    __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << idx_0 << "ul ) );\n";

                // gather values (send i1 values to respective i0 positions)
                std::uint64_t idx_j = 0;
                idx_j += std::uint64_t( op0_i1 ) << 8 * op0_i0;
                idx_j += std::uint64_t( op1_i1 ) << 8 * op1_i0;
                code << "    __m512i idx_j = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << idx_j << "ul ) );\n";

                code << "    __m512d di_j = _mm512_permutex2var_pd( di_0, idx_j, di_0 );\n";
                code << "    __m512d px_j = _mm512_permutex2var_pd( px_0, idx_j, px_0 );\n";
                code << "    __m512d py_j = _mm512_permutex2var_pd( py_0, idx_j, py_0 );\n";

                int mask = 0;
                mask += 1 << op0_i0;
                mask += 1 << op1_i0;

                // code << "         __m512d m = _mm512_div_pd( di_0, _mm512_sub_pd( di_j, di_0 ) );\n";
                code << "    __m512d m = _mm512_maskz_div_pd( " << std::dec << mask << ", di_0, _mm512_sub_pd( di_j, di_0 ) );\n";
                code << "    __m512d inter_x = _mm512_sub_pd( px_0, _mm512_mul_pd( m, _mm512_sub_pd( px_j, px_0 ) ) );\n";
                code << "    __m512d inter_y = _mm512_sub_pd( py_0, _mm512_mul_pd( m, _mm512_sub_pd( py_j, py_0 ) ) );\n";
            } else {
                // make a `idx_0` (for _mm512_permutex2var_pd)
                std::uint64_t idx_0 = 0;
                for( std::size_t i = 0; i < interpolation_indices.size(); ++i )
                    idx_0 += std::uint64_t( 8 + i ) << 8 * interpolation_indices[ i ];
                for( std::size_t i : permutation_indices )
                    idx_0 += std::uint64_t( ops[ i ].i0 ) << 8 * i;
                code << "    __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << idx_0 << std::dec << "ul ) );\n";

                // gather values
                auto set_pd = [&]( std::string out, std::string inp, int i0, int i1 ) {
                    if ( i0 != i1 )
                        code << "    __m128d " << out << " = _mm_set_pd( " << inp << "_" << i1 << ", " << inp << "_" << i0 << " );\n";
                    else
                        code << "    __m128d " << out << " = _mm_set1_pd( " << inp << "_" << i0 << " );\n";
                };

                std::size_t it_0 = interpolation_indices[ 0 ];
                std::size_t it_1 = interpolation_indices[ 1 ];
                const Op &op_0 = ops[ it_0 ];
                const Op &op_1 = ops[ it_1 ];

                set_pd( "d_i0", "d", op_0.i0, op_1.i0 );
                set_pd( "x_i0", "x", op_0.i0, op_1.i0 );
                set_pd( "y_i0", "y", op_0.i0, op_1.i0 );

                set_pd( "d_i1", "d", op_0.i1, op_1.i1 );
                set_pd( "x_i1", "x", op_0.i1, op_1.i1 );
                set_pd( "y_i1", "y", op_0.i1, op_1.i1 );

                code << "    __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );\n";
                code << "    __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );\n";
                code << "    __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );\n";

                code << "    __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( ";
                disp_c( op_1 );
                code << ", ";
                disp_c( op_0 );
                code << " ) );\n";
            }
        } else {
            // make a `idx_0` (for _mm512_permutex2var_pd)
            std::uint64_t idx_0 = 0;
            for( std::size_t i = 0; i < interpolation_indices.size(); ++i )
                idx_0 += ( std::uint64_t( 8 + i ) << 8 * interpolation_indices[ i ] );
            for( std::size_t i : permutation_indices )
                idx_0 += std::uint64_t( ops[ i ].i0 ) << 8 * i;
            code << "    __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << idx_0 << std::dec << "ul ) );\n";

            // compute the new values
            ASSERT( interpolation_indices.size() == 1, "weird" );
            std::size_t it_0 = interpolation_indices[ 0 ];
            const Op &op_0 = ops[ it_0 ];

            code << "    TF m = d_" << op_0.i0 << " / ( d_" << op_0.i1 << " - d_" << op_0.i0 << " ); // 1\n";
            code << "    __m512d inter_x = _mm512_set1_pd( x_" << op_0.i0 << " - m * ( x_" << op_0.i1 << " - x_" << op_0.i0 << " ) );\n";
            code << "    __m512d inter_y = _mm512_set1_pd( y_" << op_0.i0 << " - m * ( y_" << op_0.i1 << " - y_" << op_0.i0 << " ) );\n";
            code << "    __m512i inter_c = _mm512_set1_epi64( ";
            disp_c( op_0 );
            code << " );\n";
        }

        code << "    px = px.permute( px, idx_0, inter_x );\n";
        code << "    py = _mm512_permutex2var_pd( py, idx_0, inter_y );\n";
        code << "    pc = _mm512_permutex2var_epi64( pc, idx_0, inter_c );\n";
    }

    void write_code_scalar( std::ostream &code, int simd_size, std::size_t beg = 0 ) {
        auto x = [&]( int i ) { return i < simd_size ? "px[ " + std::to_string( i ) + " ]" : "nodes.xs[ "      + std::to_string( i ) + " ]"; };
        auto y = [&]( int i ) { return i < simd_size ? "py[ " + std::to_string( i ) + " ]" : "nodes.ys[ "      + std::to_string( i ) + " ]"; };
        auto c = [&]( int i ) { return i < simd_size ? "pc[ " + std::to_string( i ) + " ]" : "nodes.cut_ids[ " + std::to_string( i ) + " ]"; };

        // ratios
        for( std::size_t i = beg; i < ops.size(); ++i ) {
            const Op &op = ops[ i ];
            if ( op.i1 >= 0 ) {
                code << "    TF m_" << op.i0 << "_" << op.i1 << " = d_" << op.i0 << " / ( d_" << op.i1 << " - d_" << op.i0 << " );\n";
            }
        }

        // coordinates
        for( std::size_t i = beg; i < ops.size(); ++i ) {
            const Op &op = ops[ i ];
            if ( op.i1 >= 0 ) {
                code << "    TF nx_" << i << " = " << x( op.i0 ) << " - m_" << op.i0 << "_" << op.i1 << " * ( " << x( op.i1 ) << " - " << x( op.i0 ) << " );\n";
                code << "    TF ny_" << i << " = " << y( op.i0 ) << " - m_" << op.i0 << "_" << op.i1 << " * ( " << y( op.i1 ) << " - " << y( op.i0 ) << " );\n";
                code << "    TF nc_" << i << " = cut_id[ num_cut ];\n";
            } else if ( op.i0 != int( i ) ) {
                code << "    TF nx_" << i << " = " << x( op.i0 ) << ";\n";
                code << "    TF ny_" << i << " = " << y( op.i0 ) << ";\n";
                code << "    TF nc_" << i << " = " << c( op.i0 ) << ";\n";
            }
        }

        // save
        for( std::size_t i = beg; i < ops.size(); ++i ) {
            const Op &op = ops[ i ];
            if ( op.i1 >= 0 || op.i0 != int( i ) ) {
                code << "    " << x( i ) << " = nx_" << i << ";\n";
                code << "    " << y( i ) << " = ny_" << i << ";\n";
                code << "    " << c( i ) << " = nc_" << i << ";\n";
            }
        }
    }

    void write_code( std::ostream &code, int simd_size ) {
        if ( ops.size() != old_size )
            code << "    nodes_size = " << ops.size() << ";\n";

        // nothing to change excepted the size ?
        bool nothing_to_change = true;
        for( std::size_t i = 0; i < ops.size(); ++i ) {
            if ( ops[ i ].i1 >= 0 || ops[ i ].i0 != int( i ) ) {
                nothing_to_change = false;
                break;
            }
        }
        if ( nothing_to_change )
            return;

        // needed values and distances
        std::set<int> needed_val_indices, needed_distances;
        for( int i = 0; i < int( ops.size() ); ++i ) {
            const Op &op = ops[ i ];
            if ( op.i1 >= 0 ) {
                needed_val_indices.insert( op.i0 );
                needed_val_indices.insert( op.i1 );
                needed_distances.insert( op.i0 );
                needed_distances.insert( op.i1 );
            } else if ( i >= simd_size && i != op.i0 ) {
                needed_val_indices.insert( op.i0 );
            }
        }
        for( int nd : needed_distances )
            code << "    TF d_" << nd << " = di[ " << nd << " ];\n";
        for( int nd : needed_val_indices )
            code << "    TF x_" << nd << " = px[ " << nd << " ];\n";
        for( int nd : needed_val_indices )
            code << "    TF y_" << nd << " = py[ " << nd << " ];\n";

        // bulk
        //        write_code_mm128( code, simd_size );
        //        write_code_scalar( code, simd_size );
        write_code_scalar( code, simd_size, 0 );
    }


    std::size_t old_size;
    std::vector<Op> ops;
};


bool get_code( std::ostringstream &code, int nodes_size, std::bitset<8> outside, int simd_size ) {
    const int nb_outside = outside.count();

    // nothing to change
    if ( nb_outside == 0 ) {
        code << "    continue;\n";
        return true;
    }

    // totally outside
    if ( nodes_size <= 2 || nb_outside == nodes_size ) {
        if ( nodes_size )
            code << "    nodes_size = 0;\n";
        code << "    break;\n";
        return true;
    }

    // get list of modifications
    Mod mod;
    mod.old_size = nodes_size;
    for( int i = 0; i < nodes_size; ++i ) {
        int h = ( i + nodes_size - 1 ) % nodes_size;
        int j = ( i              + 1 ) % nodes_size;

        // inside point => we keep it
        if ( outside[ i ] == 0 ) {
            mod.ops.push_back( { i, -1, 0 } );
            continue;
        }

        // outside point => create points on boundaries
        if ( ! outside[ h ] )
            mod.ops.push_back( { i, h, +1 } );
        if ( ! outside[ j ] )
            mod.ops.push_back( { i, j, -1 } );
    }
    mod.find_best_rotation();

    // to lighten the generated code, we remove the rare cases
    int nb_interp = 0;
    for( Op op : mod.ops )
        nb_interp += op.i1 >= 0;
    if ( nb_interp > 2 )
        return false;

    // write code
    code << "    // size=" << nodes_size << " outside=" << outside << " mod=" << mod << "\n";
    mod.write_code( code, simd_size );

    //
    if ( int( mod.ops.size() ) <= simd_size )
        code << "    continue;\n";
    else
        code << "    break;\n";
    return true;
}

void generate( int simd_size = 8 ) {
    std::map<std::string,int> code_map; // code content => goto number
    std::vector<int> case_nums; // outside_nodes code => code number
    case_nums.resize( 1 << ( simd_size + 1 ), 0 );
    code_map[ "" ] = 0;

    // make the cases
    for( int nb_nodes = 0; nb_nodes <= simd_size; ++nb_nodes ) {
        for( int outside_case = 0; outside_case < ( 1 << simd_size ); ++outside_case ) {
            std::ostringstream code;
            if ( get_code( code, nb_nodes, outside_case, simd_size ) ) {
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

    // jump code
    std::cout << "static void *dispatch_table[] = {";
    for( std::size_t n = 0; n < case_nums.size(); ++n )
        std::cout << ( n % 16 ? " " : "\n    " ) << "&&case_" << case_nums[ n ] << ",";
    std::cout << "};\n";
    std::cout << "goto *dispatch_table[ case_code ];\n";

    // cases code
    for( auto iter : code_map ) {
        // the generic case is manually written
        if ( iter.second == 0 )
            continue;
        std::cout << "case_" << iter.second << ": {\n";
        std::cout << iter.first;
        std::cout << "}\n";
    }
    std::cout << "case_0:\n";
    std::cout << "    // generic case\n";
    std::cout << "    ;\n";
}

int main() {
    generate( 8 );
}


