#include "../../Support/Display/generic_ostream_output.h"
#include "../../Support/bit_handling.h"
#include "../../Support/TODO.h"
#include <iostream>
#include <sstream>
#include <vector>
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

    void write_code( std::ostream &code, int simd_size, int nb_regs, bool scalar = false ) {
        // get the needed distances
        std::set<int> needed_distances;
        for( std::size_t i = 0; i < ops.size(); ++i ) {
            const Op &op = ops[ i ];
            if ( op.i1 >= 0 ) {
                needed_distances.insert( op.i0 );
                needed_distances.insert( op.i1 );
            }
        }
        for( int nd : needed_distances )
            code << "            TF d_" << nd << " = reinterpret_cast<const TF *>( &di_" << nd / simd_size << " )[ " << nd % simd_size << " ];\n";

        if ( scalar ) {
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
        } else {
            // Prop: on fait des permutations
            bool has_permutations = false, has_interpolations = false;
            for( std::size_t i = 0; i < ops.size(); ++i ) {
                if ( ops[ i ].i1 < 0 && ops[ i ].i0 != int( i ) )
                    has_permutations = true;
                if ( ops[ i ].i1 >= 0 )
                    has_interpolations = true;
            }

            // _mm512_permutex2var_pd
            if ( has_permutations || has_interpolations ) {
                int num_interpolation = 0;
                std::uint64_t permutations = 0;
                for( std::size_t i = 0; i < ops.size(); ++i ) {
                    const Op &op = ops[ i ];

                    // if it does not fit into the registers, make intermediate variables
                    if ( int( i ) >= nb_regs * simd_size ) {
                        if ( op.i1 >= 0 ) {
                            code << "            TF m_" << op.i0 << "_" << op.i1 << " = d_" << op.i0 << " / ( d_" << op.i1 << " - d_" << op.i0 << " );\n";
                            code << "            TF x_" << i << " = x[ " << op.i0 << " ] - m_" << op.i0 << "_" << op.i1 << " * ( x[ " << op.i1 << " ] - x[ " << op.i0 << " ] );\n";
                            code << "            TF y_" << i << " = y[ " << op.i0 << " ] - m_" << op.i0 << "_" << op.i1 << " * ( y[ " << op.i1 << " ] - y[ " << op.i0 << " ] );\n";
                        } else if ( op.i0 != int( i ) ) {
                            code << "            TF x_" << i << " = x[ " << op.i0 << " ];\n";
                            code << "            TF y_" << i << " = y[ " << op.i0 << " ];\n";
                        }
                    } else if ( op.i1 >= 0 ) { // variables for interpolation
                        code << "            TF m_" << num_interpolation << " = d_" << op.i0 << " / ( d_" << op.i1 << " - d_" << op.i0 << " );\n";
                        code << "            TF x_" << num_interpolation << " = reinterpret_cast<double *>( &px_" << op.i0 / simd_size << " )[ " << op.i0 % simd_size << " ] - m_" << num_interpolation << " * ( reinterpret_cast<double *>( &px_" << op.i1 / simd_size << " )[ " << op.i1 % simd_size << " ] - reinterpret_cast<double *>( &px_" << op.i0 / simd_size << " )[ " << op.i0 % simd_size << " ] );\n";
                        code << "            TF y_" << num_interpolation << " = reinterpret_cast<double *>( &py_" << op.i0 / simd_size << " )[ " << op.i0 % simd_size << " ] - m_" << num_interpolation << " * ( reinterpret_cast<double *>( &py_" << op.i1 / simd_size << " )[ " << op.i1 % simd_size << " ] - reinterpret_cast<double *>( &py_" << op.i0 / simd_size << " )[ " << op.i0 % simd_size << " ] );\n";

                        permutations += std::uint64_t( 8 + num_interpolation++ ) << 8 * i;
                    } else
                        permutations += std::uint64_t( op.i0 ) << 8 * i;
                }

                // save value outside the registers
                for( std::size_t i = 0; i < ops.size(); ++i ) {
                    const Op &op = ops[ i ];
                    if ( int( i ) >= nb_regs * simd_size && ( op.i1 >= 0 || op.i0 != int( i ) ) ) {
                        code << "            x[ " << i << " ] = x_" << i << ";\n";
                        code << "            y[ " << i << " ] = y_" << i << ";\n";
                    }
                }

                if ( nb_regs > 1 )
                    TODO;
                code << "            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x" << std::hex << permutations << "ul ) );\n";

                if ( num_interpolation ) {
                    if ( num_interpolation == 1 ) {
                        code << "            __m512d inter_x = _mm512_set1_pd( x_0 );\n";
                        code << "            __m512d inter_y = _mm512_set1_pd( y_0 );\n";
                    } else if ( num_interpolation == 2 ) {
                        code << "            __m512d inter_x = _mm512_castpd128_pd512( _mm_set_pd( x_1, x_0 ) );\n";
                        code << "            __m512d inter_y = _mm512_castpd128_pd512( _mm_set_pd( y_1, y_0 ) );\n";
                    } else {
                        std::cerr << num_interpolation << std::endl;
                        TODO;
                    }
                    code << "            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );\n";
                    code << "            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );\n";
                } else {
                    code << "            px_0 = _mm512_permutex2var_pd( px_0, idx_0, px_0 );\n";
                    code << "            py_0 = _mm512_permutex2var_pd( py_0, idx_0, py_0 );\n";
                }

                //                if ( ops.size() == 4 && old_size == 4 ) {
                //                    code << "            P( reinterpret_cast<std::uint64_t *>( &idx_0 )[ 0 ], reinterpret_cast<std::uint64_t *>( &idx_0 )[ 1 ], reinterpret_cast<std::uint64_t *>( &idx_0 )[ 2 ], reinterpret_cast<std::uint64_t *>( &idx_0 )[ 3 ] );\n";
                //                    code << "            P( reinterpret_cast<double *>( &inter_x )[ 0 ], reinterpret_cast<double *>( &inter_x )[ 1 ] );\n";
                //                    code << "            P( reinterpret_cast<double *>( &px_0 )[ 0 ], reinterpret_cast<double *>( &px_0 )[ 1 ], reinterpret_cast<double *>( &px_0 )[ 2 ], reinterpret_cast<double *>( &px_0 )[ 3 ] );\n";
                //                }
            }
        }

        if ( ops.size() != old_size )
            code << "            size = " << ops.size() << ";\n";
        code << "            break;\n";
    }

    std::size_t old_size;
    std::vector<Op> ops;
};


void get_code( std::ostringstream &code, int index, int max_size_included, int simd_size ) {
    int nb_regs = ( max_size_included + simd_size - 1 ) / simd_size;
    const int mul_size = 1 << max_size_included;
    const int size = index / mul_size;

    std::bitset<64> outside = index & ( ( 1 << size ) - 1 );
    const int nb_outside = sdot::popcnt( outside );

    if ( size <= 2 || nb_outside == size ) {
        if ( size )
            code << "            size = 0;\n";
        code << "            break; // totally outside\n";
        return;
    }

    if ( nb_outside == 0 ) {
        code << "            break;\n";
        return;
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
    if ( nb_interp > 2 ) {
        code << "        plane_cut_gen( cut, N<flags>() );\n";
        for( int i = 0; i < nb_regs; ++i ) {
            code << "        px_" << i << " = _mm512_load_pd( x + " << simd_size * i << " );\n";
            code << "        py_" << i << " = _mm512_load_pd( y + " << simd_size * i << " );\n";
        }
        return;
    }

    // write code
    code << "            // size=" << size << " outside=" << outside << " mod=" << mod << "\n";
    mod.write_code( code, simd_size, nb_regs );
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
    std::cout << "        const Cut &cut = cuts[ num_cut ];\n";
    std::cout << "        __m512d rd = _mm512_set1_pd( cut.dist );\n";
    std::cout << "        __m512d nx = _mm512_set1_pd( cut.dir.x );\n";
    std::cout << "        __m512d ny = _mm512_set1_pd( cut.dir.y );\n";
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
        get_code( code, index, max_size_included, simd_size );
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
    std::cout << "          plane_cut_gen( cut, N<flags>() );\n";
    for( int i = 0; i < nb_regs; ++i ) {
        std::cout << "          px_" << i << " = _mm512_load_pd( x + " << simd_size * i << " );\n";
        std::cout << "          py_" << i << " = _mm512_load_pd( y + " << simd_size * i << " );\n";
    }
    std::cout << "          break;\n";
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
    std::cout << "#define MM256_SET_PD( A, B, C, D ) _mm256_set_pd( D, C, B, A ) \n";
    std::cout << "#define MM128_SET_PD( A, B ) _mm_set_pd( B, A ) \n";
    std::cout << "namespace sdot {\n";

    //
    std::cout << "\n";
    std::cout << "template<class Pc> template<int flags>\n";
    std::cout << "void ConvexPolyhedron2<Pc>::plane_cut_simd_switch( const Cut *cuts, std::size_t nb_cuts, N<flags>, S<double>, S<std::uint64_t> ) {\n";
    std::cout << "    #ifdef __AVX512F__\n";
    generate( 8, "AVX512", 8 );
    std::cout << "    #else // __AVX512F__\n";
    std::cout << "    for( std::size_t i = 0; i < nb_cuts; ++i )\n";
    std::cout << "        plane_cut_gen( cuts[ i ], N<flags>() );\n";
    std::cout << "    #endif // __AVX512F__\n";
    std::cout << "}\n";

    // generic version
    std::cout << "\n";
    std::cout << "template<class Pc> template<int flags,class T,class U>\n";
    std::cout << "void ConvexPolyhedron2<Pc>::plane_cut_simd_switch( const Cut *cuts, std::size_t nb_cuts, N<flags>, S<T>, S<U> ) {\n";
    std::cout << "    for( std::size_t i = 0; i < nb_cuts; ++i )\n";
    std::cout << "        plane_cut_gen( cuts[ i ], N<flags>() );\n";
    std::cout << "}\n";

    std::cout << "\n";
    std::cout << "} // namespace sdot\n";
}


