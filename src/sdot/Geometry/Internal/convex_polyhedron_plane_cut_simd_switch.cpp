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

    int i0; ///<
    int i1; ///< -1 means we take the values only from i0
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

    void write_code_mm256xy( std::ostream &code, int simd_size, int nb_regs ) {
        // get op indices for each type of operation
        std::vector<std::size_t> permutation_indices, interpolation_indices;
        for( std::size_t i = 0; i < std::min( ops.size(), std::size_t( nb_regs * simd_size ) ); ++i ) {
            if ( ops[ i ].i1 >= 0 )
                interpolation_indices.push_back( i );
            else
                permutation_indices.push_back( i );
        }

        // make `idx_0` and `idy_0` (for _mm512_permutex2var_pd)
        std::uint64_t idx_0 = 0, idy_0 = 0;
        for( std::size_t i = 0; i < interpolation_indices.size(); ++i ) {
            idx_0 += ( std::uint64_t( 8 + 2 * i ) << 8 * interpolation_indices[ i ] );
            idy_0 += ( std::uint64_t( 9 + 2 * i ) << 8 * interpolation_indices[ i ] );
        }
        for( std::size_t i : permutation_indices ) {
            idx_0 += std::uint64_t( ops[ i ].i0 ) << 8 * i;
            idy_0 += std::uint64_t( ops[ i ].i0 ) << 8 * i;
        }
        code << "            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << idx_0 << "ul ) );\n";
        code << "            __m512i idy_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << idy_0 << "ul ) );\n";
        if ( nb_regs > 1 )
            TODO;

        if ( interpolation_indices.size() == 2 ) {
            auto set_pd = [&]( std::string out, std::string x, std::string y, int i0, int i1 ) {
                if ( i0 == i1 && x == y )
                    code << "            __m256d " << out << " = _mm256_set1_pd( " << x << "_" << i0 << " );\n";
                else if ( x == y )
                    code << "            __m256d " << out << " = _mm256_set_pd( " << y << "_" << i1 << ", " << x << "_" << i1 << ", " << y << "_" << i0 << ", " << x << "_" << i0 << " );\n";
                else
                    code << "            __m256d " << out << " = _mm256_set_pd( " << y << "_" << i1 << ", " << x << "_" << i1 << ", " << y << "_" << i0 << ", " << x << "_" << i0 << " );\n";
            };

            std::size_t it_0 = interpolation_indices[ 0 ];
            std::size_t it_1 = interpolation_indices[ 1 ];
            const Op &op_0 = ops[ it_0 ];
            const Op &op_1 = ops[ it_1 ];

            set_pd( "d_i0", "d", "d", op_0.i0, op_1.i0 );
            set_pd( "z_i0", "x", "y", op_0.i0, op_1.i0 );

            set_pd( "d_i1", "d", "d", op_0.i1, op_1.i1 );
            set_pd( "z_i1", "x", "y", op_0.i1, op_1.i1 );

            code << "            __m256d m = _mm256_div_pd( d_i0, _mm256_sub_pd( d_i1, d_i0 ) );\n";
            code << "            __m512d inter_z = _mm512_castpd256_pd512( _mm256_sub_pd( z_i0, _mm256_mul_pd( m, _mm256_sub_pd( z_i1, z_i0 ) ) ) );\n";
        } else {
            ASSERT( interpolation_indices.size() == 1, "weird" );
            std::size_t it_0 = interpolation_indices[ 0 ];
            const Op &op_0 = ops[ it_0 ];

            code << "            TF m = d_" << op_0.i0 << " / ( d_" << op_0.i1 << " - d_" << op_0.i0 << " ); // 1\n";
            code << "            __m512d inter_z = _mm512_castpd128_pd512( _mm_set_pd( y_" << op_0.i0 << " - m * ( y_" << op_0.i1 << " - y_" << op_0.i0 << " ), x_" << op_0.i0 << " - m * ( x_" << op_0.i1 << " - x_" << op_0.i0 << " ) ) );\n";
        }

        code << "            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_z );\n";
        code << "            py_0 = _mm512_permutex2var_pd( py_0, idy_0, inter_z );\n";
    }

    void write_code_mm128( std::ostream &code, int simd_size, int nb_regs, bool inplace = false ) {
        // get op indices for each type of operation
        std::vector<std::size_t> permutation_indices, interpolation_indices;
        for( std::size_t i = 0; i < std::min( ops.size(), std::size_t( nb_regs * simd_size ) ); ++i ) {
            if ( ops[ i ].i1 >= 0 )
                interpolation_indices.push_back( i );
            else
                permutation_indices.push_back( i );
        }
        if ( nb_regs > 1 )
            TODO;

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
                code << "            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << idx_0 << "ul ) );\n";

                // gather values (send i1 values to respective i0 positions)
                std::uint64_t idx_j = 0;
                idx_j += std::uint64_t( op0_i1 ) << 8 * op0_i0;
                idx_j += std::uint64_t( op1_i1 ) << 8 * op1_i0;
                code << "            __m512i idx_j = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << idx_j << "ul ) );\n";

                code << "            __m512d di_j = _mm512_permutex2var_pd( di_0, idx_j, di_0 );\n";
                code << "            __m512d px_j = _mm512_permutex2var_pd( px_0, idx_j, px_0 );\n";
                code << "            __m512d py_j = _mm512_permutex2var_pd( py_0, idx_j, py_0 );\n";

                int mask = 0;
                mask += 1 << op0_i0;
                mask += 1 << op1_i0;

                // code << "            __m512d m = _mm512_div_pd( di_0, _mm512_sub_pd( di_j, di_0 ) );\n";
                code << "            __m512d m = _mm512_maskz_div_pd( " << std::dec << mask << ", di_0, _mm512_sub_pd( di_j, di_0 ) );\n";
                code << "            __m512d inter_x = _mm512_sub_pd( px_0, _mm512_mul_pd( m, _mm512_sub_pd( px_j, px_0 ) ) );\n";
                code << "            __m512d inter_y = _mm512_sub_pd( py_0, _mm512_mul_pd( m, _mm512_sub_pd( py_j, py_0 ) ) );\n";
            } else {
                // make a `idx_0` (for _mm512_permutex2var_pd)
                std::uint64_t idx_0 = 0;
                for( std::size_t i = 0; i < interpolation_indices.size(); ++i )
                    idx_0 += std::uint64_t( 8 + i ) << 8 * interpolation_indices[ i ];
                for( std::size_t i : permutation_indices )
                    idx_0 += std::uint64_t( ops[ i ].i0 ) << 8 * i;
                code << "            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << idx_0 << std::dec << "ul ) );\n";

                // gather values
                auto set_pd = [&]( std::string out, std::string inp, int i0, int i1 ) {
                    if ( i0 != i1 )
                        code << "            __m128d " << out << " = _mm_set_pd( " << inp << "_" << i1 << ", " << inp << "_" << i0 << " );\n";
                    else
                        code << "            __m128d " << out << " = _mm_set1_pd( " << inp << "_" << i0 << " );\n";
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

                code << "            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );\n";
                code << "            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );\n";
                code << "            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );\n";
            }
        } else {
            // make a `idx_0` (for _mm512_permutex2var_pd)
            std::uint64_t idx_0 = 0;
            for( std::size_t i = 0; i < interpolation_indices.size(); ++i )
                idx_0 += ( std::uint64_t( 8 + i ) << 8 * interpolation_indices[ i ] );
            for( std::size_t i : permutation_indices )
                idx_0 += std::uint64_t( ops[ i ].i0 ) << 8 * i;
            code << "            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << idx_0 << std::dec << "ul ) );\n";

            // compute the new values
            ASSERT( interpolation_indices.size() == 1, "weird" );
            std::size_t it_0 = interpolation_indices[ 0 ];
            const Op &op_0 = ops[ it_0 ];

            code << "            TF m = d_" << op_0.i0 << " / ( d_" << op_0.i1 << " - d_" << op_0.i0 << " ); // 1\n";
            code << "            __m512d inter_x = _mm512_set1_pd( x_" << op_0.i0 << " - m * ( x_" << op_0.i1 << " - x_" << op_0.i0 << " ) );\n";
            code << "            __m512d inter_y = _mm512_set1_pd( y_" << op_0.i0 << " - m * ( y_" << op_0.i1 << " - y_" << op_0.i0 << " ) );\n";
        }

        code << "            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );\n";
        code << "            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );\n";
    }

    void write_code_scalar( std::ostream &code ) {
        // ratios
        for( std::size_t i = 0; i < ops.size(); ++i ) {
            const Op &op = ops[ i ];
            if ( op.i1 >= 0 ) {
                code << "            TF m_" << op.i0 << "_" << op.i1 << " = d_" << op.i0 << " / ( d_" << op.i1 << " - d_" << op.i0 << " );\n";
            }
        }

        // coordinates
        for( std::size_t i = 0; i < ops.size(); ++i ) {
            const Op &op = ops[ i ];
            if ( op.i1 >= 0 ) {
                code << "            TF x_" << i << " = x[ " << op.i0 << " ] - m_" << op.i0 << "_" << op.i1 << " * ( x[ " << op.i1 << " ] - x[ " << op.i0 << " ] );\n";
                code << "            TF y_" << i << " = y[ " << op.i0 << " ] - m_" << op.i0 << "_" << op.i1 << " * ( y[ " << op.i1 << " ] - y[ " << op.i0 << " ] );\n";
            } else if ( op.i0 != int( i ) ) {
                code << "            TF x_" << i << " = x[ " << op.i0 << " ];\n";
                code << "            TF y_" << i << " = y[ " << op.i0 << " ];\n";
            }
        }

        // save
        for( std::size_t i = 0; i < ops.size(); ++i ) {
            const Op &op = ops[ i ];
            if ( op.i1 >= 0 || op.i0 != int( i ) ) {
                code << "            x[ " << i << " ] = x_" << i << ";\n";
                code << "            y[ " << i << " ] = y_" << i << ";\n";
            }
        }
    }

    void write_code( std::ostream &code, int simd_size, int nb_regs, bool scalar = false ) {
        if ( ops.size() != old_size )
            code << "            size = " << ops.size() << ";\n";

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
        for( std::size_t i = 0; i < ops.size(); ++i ) {
            const Op &op = ops[ i ];
            if ( op.i1 >= 0 ) {
                needed_val_indices.insert( op.i0 );
                needed_val_indices.insert( op.i1 );
                needed_distances.insert( op.i0 );
                needed_distances.insert( op.i1 );
            } else if ( ( scalar || int( i ) >= nb_regs * simd_size ) && int( i ) != op.i0 ) {
                needed_val_indices.insert( op.i0 );
            }
        }
        for( int nd : needed_distances )
            code << "            TF d_" << nd << " = reinterpret_cast<const TF *>( &di_" << nd / simd_size << " )[ " << nd % simd_size << " ];\n";
        for( int nd : needed_val_indices )
            code << "            TF x_" << nd << " = reinterpret_cast<const TF *>( &px_" << nd / simd_size << " )[ " << nd % simd_size << " ];\n";
        for( int nd : needed_val_indices )
            code << "            TF y_" << nd << " = reinterpret_cast<const TF *>( &py_" << nd / simd_size << " )[ " << nd % simd_size << " ];\n";

        // no simd ?
        if ( scalar )
            return write_code_scalar( code );

        // write values outside the registers
        for( std::size_t i = nb_regs * simd_size; i < ops.size(); ++i ) {
            int i0 = ops[ i ].i0, i1 = ops[ i ].i1;
            if ( i1 >= 0 ) {
                code << "            TF m_" << i0 << "_" << i1 << " = d_" << i0 << " / ( d_" << i1 << " - d_" << i0 << " );\n";
                code << "            x[ " << i << " ] = x_" << i0 << " - m_" << i0 << "_" << i1 << " * ( x_" << i1 << " - x_" << i0 << " );\n";
                code << "            y[ " << i << " ] = y_" << i0 << " - m_" << i0 << "_" << i1 << " * ( y_" << i1 << " - y_" << i0 << " );\n";
            } else if ( i0 != int( i ) ) {
                code << "            x[ " << i << " ] = x_" << i0 << ";\n";
                code << "            y[ " << i << " ] = y_" << i0 << ";\n";
            }
        }

        // bulk
        write_code_mm128( code, simd_size, nb_regs );
        // write_code_mm256xy( code, simd_size, nb_regs );

        code << "            break;\n";
    }


    std::size_t old_size;
    std::vector<Op> ops;
};


bool get_code( std::ostringstream &code, int index, int max_size_included, int simd_size ) {
    int nb_regs = ( max_size_included + simd_size - 1 ) / simd_size;
    const int mul_size = 1 << max_size_included;
    const int size = index / mul_size;

    std::bitset<64> outside = index & ( ( 1 << size ) - 1 );
    const int nb_outside = sdot::popcnt( outside );

    // nothing to change
    if ( nb_outside == 0 ) {
        code << "            break;\n";
        return true;
    }

    // totally outside
    if ( size <= 2 || nb_outside == size ) {
        if ( size )
            code << "            size = 0;\n";
        code << "            break;\n";
        return true;
    }

    // get list of modifications
    Mod mod;
    mod.old_size = size;
    for( int i = 0; i < size; ++i ) {
        int h = ( i + size - 1 ) % size;
        int j = ( i + 1 ) % size;

        // inside point => we keep it
        if ( outside[ i ] == 0 ) {
            mod.ops.push_back( { i, -1 } );
            continue;
        }

        // outside point => create points on boundaries
        if ( ! outside[ h ] )
            mod.ops.push_back( { i, h } );
        if ( ! outside[ j ] )
            mod.ops.push_back( { i, j } );
    }
    mod.find_best_rotation();

    // to lighten the generated code, we remove the rare cases
    int nb_interp = 0;
    for( Op op : mod.ops )
        nb_interp += op.i1 >= 0;
    if ( nb_interp > 2 )
        return false;

    // write code
    code << "            // size=" << size << " outside=" << outside << " mod=" << mod << "\n";
    mod.write_code( code, simd_size, nb_regs );
    return true;
}

void generate( int simd_size, std::string /*ext*/, int max_size_included = 8 ) {
    int mul_size = 1 << max_size_included;
    int max_index = mul_size * ( max_size_included + 1 );
    int nb_regs = ( max_size_included + simd_size - 1 ) / simd_size;

    std::cout << "    // outsize list\n";
    std::cout << "    TF *x = &nodes->x;\n";
    std::cout << "    TF *y = &nodes->y;\n";
    for( int i = 0; i < nb_regs; ++i ) {
        std::cout << "    __m512d px_" << i << " = _mm512_load_pd( x + " << simd_size * i << " );\n";
        std::cout << "    __m512d py_" << i << " = _mm512_load_pd( y + " << simd_size * i << " );\n";
    }
    std::cout << "    for( std::size_t num_cut = 0; num_cut < nb_cuts; ++num_cut ) {\n";
    std::cout << "        __m512d rd = _mm512_set1_pd( cut_ps[ num_cut ] );\n";
    std::cout << "        __m512d nx = _mm512_set1_pd( cut_dx[ num_cut ] );\n";
    std::cout << "        __m512d ny = _mm512_set1_pd( cut_dy[ num_cut ] );\n";
    for( int i = 0; i < nb_regs; ++i ) {
        std::cout << "        __m512d bi_" << i << " = _mm512_add_pd( _mm512_mul_pd( px_" << i << ", nx ), _mm512_mul_pd( py_" << i << ", ny ) );\n";
        std::cout << "        std::uint8_t outside_" << i << " = _mm512_cmp_pd_mask( bi_" << i << ", rd, _CMP_GT_OQ );\n"; // OQ => 46.9, QS => 47.1
        std::cout << "        __m512d di_" << i << " = _mm512_sub_pd( bi_" << i << ", rd );\n";
        // std::cout << "        outside_" << i << " &= ( 1 << size ) - 1;\n";
        // std::cout << "        std::uint8_t outside_" << i << " = _mm512_movepi64_mask( __m512i( di_" << i << " ) );\n"; // => 47.1
    }
    std::cout << "\n";

    // gather
    std::map<std::string,std::vector<int>> cases;
    for( int index = 0; index < max_index; ++index ) {
        std::ostringstream code;
        if ( get_code( code, index, max_size_included, simd_size ) )
            cases[ code.str() ].push_back( index );
    }

    // cases
    std::cout << "        switch( " << mul_size << " * size + ";
    for( int i = 0; i < nb_regs; ++i )
        std::cout << ( 1 << ( i * simd_size ) ) <<  " * outside_" << i;
    std::cout << " ) {";
    for( std::pair<std::string,std::vector<int>> c : cases ) {
        for( int index : c.second )
            std::cout << "\n        case " << index << ":";
        std::cout << " {\n" << c.first << "        }";
    }
    std::cout << "\n        default:\n";
    for( int i = 0; i < nb_regs; ++i ) {
        std::cout << "            _mm512_store_pd( x + " << simd_size * i << ", px_" << i << " );\n";
        std::cout << "            _mm512_store_pd( y + " << simd_size * i << ", py_" << i << " );\n";
    }
    std::cout << "            plane_cut_gen( cut_dx[ num_cut ], cut_dy[ num_cut ], cut_ps[ num_cut ], cut_id[ num_cut ], N<flags>() );\n";
    std::cout << "            x = &nodes->x;\n";
    std::cout << "            y = &nodes->y;\n";
    for( int i = 0; i < nb_regs; ++i ) {
        std::cout << "            px_" << i << " = _mm512_load_pd( x + " << simd_size * i << " );\n";
        std::cout << "            py_" << i << " = _mm512_load_pd( y + " << simd_size * i << " );\n";
    }
    std::cout << "            break;\n";
    std::cout << "        }\n";
    std::cout << "    }\n";

    // save regs
    for( int i = 0; i < nb_regs; ++i ) {
        std::cout << "    _mm512_store_pd( x + " << simd_size * i << ", px_" << i << " );\n";
        std::cout << "    _mm512_store_pd( y + " << simd_size * i << ", py_" << i << " );\n";
    }
}

int main() {
    std::cout << "#include \"../ConvexPolyhedron2.h\"\n";

    std::cout << "\n";
    std::cout << "namespace sdot {\n";

    // double + uint64 version
    std::cout << "\n";
    std::cout << "template<class Pc> template<int flags>\n";
    std::cout << "void ConvexPolyhedron2<Pc>::plane_cut_simd_switch( const TF *cut_dx, const TF *cut_dy, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags>, S<double>, S<std::uint64_t> ) {\n";
    std::cout << "    #ifdef __AVX512F__\n";
    generate( 8, "AVX512", 8 );
    std::cout << "    #else // __AVX512F__\n";
    std::cout << "    for( std::size_t i = 0; i < nb_cuts; ++i )\n";
    std::cout << "        plane_cut_gen( cut_dx[ i ], cut_dy[ i ], cut_ps[ i ], cut_id[ i ], N<flags>() );\n";
    std::cout << "    #endif // __AVX512F__\n";
    std::cout << "}\n";

    // generic version
    std::cout << "\n";
    std::cout << "template<class Pc> template<int flags,class T,class U>\n";
    std::cout << "void ConvexPolyhedron2<Pc>::plane_cut_simd_switch( const TF *cut_dx, const TF *cut_dy, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags>, S<T>, S<U> ) {\n";
    std::cout << "    for( std::size_t i = 0; i < nb_cuts; ++i )\n";
    std::cout << "        plane_cut_gen( cut_dx[ i ], cut_dy[ i ], cut_ps[ i ], cut_id[ i ], N<flags>() );\n";
    std::cout << "}\n";

    std::cout << "\n";
    std::cout << "} // namespace sdot\n";
}


