#include "../../Support/bit_handling.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <map>



void get_code( std::ostringstream &code, int index, int max_size_included, int simd_size, bool simd_edge_cut = true, bool loc_reg = true ) {
    int mul_size = 1 << max_size_included;

    int size = index / mul_size;
    std::bitset<64> outside = index & ( ( 1 << size ) - 1 );
    int nb_outside = sdot::popcnt( outside );

    code << "        // size=" << size << " outside=" << outside << "\n";
    // code << "        ++bc[ " << nb_outside << " ];\n";

    if ( size <= 2 || nb_outside == 0 ) {
        code << "        return false;\n";
        return;
    }

    if ( nb_outside == size ) {
        code << "        size = 0;\n";
        code << "        return true;\n";
        return;
    }

    // update normal
    // code << "        bc[ " << nb_outside << " ]++;\n";
    code << "        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )\n";
    code << "            normal /= norm_2( normal );\n";

    //
    int i1 = sdot::tzcnt( outside );

    // only 1 outside ?
    if ( nb_outside == 1 ) {
        code << "        size = " << size + 1 << ";\n";
        int i0 = ( i1 + size - 1 ) % size;
        int i2 = ( i1 + 1 ) % size;
        int in = i1 + 1;

        code << "        TF s0 = reinterpret_cast<const TF *>( &di_" << i0 / simd_size << " )[ " << i0 % simd_size << " ];\n";
        code << "        TF s1 = reinterpret_cast<const TF *>( &di_" << i1 / simd_size << " )[ " << i1 % simd_size << " ];\n";
        code << "        TF s2 = reinterpret_cast<const TF *>( &di_" << i2 / simd_size << " )[ " << i2 % simd_size << " ];\n";

        //
        //        if ( simd_edge_cut ) {
        //            code << "        Node &n0 = nodes->local_at( " << i0 << " );\n";
        //            code << "        Node &n1 = nodes->local_at( " << i1 << " );\n";
        //            code << "        Node &n2 = nodes->local_at( " << i2 << " );\n";
        //            code << "        Node &nn = nodes->local_at( " << in << " );\n";

        //            code << "        __m256d s0202 = MM256_SET_PD( s0, s2, s0, s2 );\n";
        //            code << "        __m256d s1111 = _mm256_set1_pd( s1 );\n";
        //            code << "        __m256d m0202 = s0202 / ( s1111 - s0202 );\n";

        //            code << "        __m256d n02xy = MM256_SET_PD( n0.x, n2.x, n0.y, n2.y );\n";
        //            code << "        __m256d n11xy = MM256_SET_PD( n1.x, n1.x, n1.y, n1.y );\n";

        //            // shift points
        //            for( int i = size; i > in; --i )
        //                code << "        nodes->local_at( " << i << " ).get_straight_content_from( nodes->local_at( " << i - 1 << " ) );\n";

        //            code << "        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }\n";
        //            code << "        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }\n";
        //            code << "        nn.cut_id.set( n1.cut_id.get() );\n";
        //            code << "        n1.cut_id.set( cut_id );\n";

        //            code << "        __m256d n1nxy = n02xy - m0202 * ( n11xy - n02xy );\n";

        //            code << "        n1.x = reinterpret_cast<const double *>( &n1nxy )[ 0 ];\n";
        //            code << "        nn.x = reinterpret_cast<const double *>( &n1nxy )[ 1 ];\n";

        //            code << "        n1.y = reinterpret_cast<const double *>( &n1nxy )[ 2 ];\n";
        //            code << "        nn.y = reinterpret_cast<const double *>( &n1nxy )[ 3 ];\n";
        //            code << "        P( n1.x );\n";
        //            code << "        P( nn.x );\n";
        //            code << "        P( n1.y );\n";
        //            code << "        P( nn.y );\n";

        //            code << "        return true;\n";
        //            return;
        //        }


        // generic case (for 1 point outside)
        code << "        Node &n0 = nodes->local_at( " << i0 << " );\n";
        code << "        Node &n1 = nodes->local_at( " << i1 << " );\n";
        code << "        Node &n2 = nodes->local_at( " << i2 << " );\n";
        code << "        Node &nn = nodes->local_at( " << in << " );\n";

        code << "        TF m0 = s0 / ( s1 - s0 );\n";
        code << "        TF m1 = s2 / ( s1 - s2 );\n";

        // save coordinates that can be modified
        code << "        TF n0_x = n0.x;\n";
        code << "        TF n0_y = n0.y;\n";

        // shift points
        for( int i = size; i > in; --i )
            code << "        nodes->local_at( " << i << " ).get_straight_content_from( nodes->local_at( " << i - 1 << " ) );\n";

        code << "        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }\n";
        code << "        nn.x = n2.x - m1 * ( n1.x - n2.x );\n";
        code << "        nn.y = n2.y - m1 * ( n1.y - n2.y );\n";
        code << "        nn.cut_id.set( n1.cut_id.get() );\n";

        code << "        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }\n";
        code << "        n1.x = n0_x - m0 * ( n1.x - n0_x );\n";
        code << "        n1.y = n0_y - m0 * ( n1.y - n0_y );\n";
        code << "        n1.cut_id.set( cut_id );\n";
        code << "        return true;\n";
        return;


        //        if ( loc_reg && size < 8 && i1 % 2 == 0 ) {
        //            code << "        // loc_reg\n";
        //            code << "        TF m0 = s0 / ( s1 - s0 );\n";
        //            code << "        TF m1 = s2 / ( s1 - s2 );\n";

        //            code << "        Node &n0 = nodes->local_at( " << i0 << " );\n";
        //            code << "        Node &n1 = nodes->local_at( " << i1 << " );\n";
        //            code << "        Node &n2 = nodes->local_at( " << i2 << " );\n";
        //            code << "        Node &nn = nodes->local_at( " << in << " );\n";


        //            code << "        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }\n";
        //            code << "        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }\n";
        //            code << "        nn.cut_id.set( n1.cut_id.get() );\n";
        //            code << "        n1.cut_id.set( cut_id );\n";

        //            // get the new coordinates
        //            code << "        TF nn_x = n2.x - m1 * ( n1.x - n2.x );\n";
        //            code << "        TF nn_y = n2.y - m1 * ( n1.y - n2.y );\n";
        //            code << "        TF n1_x = n0.x - m0 * ( n1.x - n0.x );\n";
        //            code << "        TF n1_y = n0.y - m0 * ( n1.y - n0.y );\n";

        //            // expand x, y
        //            int expand_mask = 0;
        //            for( int i = 0; i <= size; ++i )
        //                if ( i != in )
        //                    expand_mask |= ( 1 << i );

        //            // __m512d _mm512_insertf64x2 (__m512d a, __m128d b, int imm8) lorsque i1 est pair
        //            code << "        px_0 = _mm512_maskz_expand_pd( " << expand_mask << ", px_0 );\n";
        //            code << "        py_0 = _mm512_maskz_expand_pd( " << expand_mask << ", py_0 );\n";

        //            code << "        px_0 = _mm512_insertf64x2( px_0, MM_SET_PD( n1_x, nn_x ), " << i1 / 2 << " );\n";
        //            code << "        py_0 = _mm512_insertf64x2( py_0, MM_SET_PD( n1_y, nn_y ), " << i1 / 2 << " );\n";

        //            code << "        _mm512_store_pd( &nodes->x, px_0 );\n";
        //            code << "        _mm512_store_pd( &nodes->y, py_0 );\n";

        //            //            for( int i = 0; i < size + 1; ++i )
        //            //                code << "        P( 'reint''erpret_cast<double *>( &px_0 )[ " << i << " ] );\n";
        //        } else {
        //            //            code << "        Node &n0 = nodes->local_at( " << i0 << " );\n";
        //            //            code << "        Node &n1 = nodes->local_at( " << i1 << " );\n";
        //            //            code << "        Node &n2 = nodes->local_at( " << i2 << " );\n";
        //            //            code << "        Node &nn = nodes->local_at( " << in << " );\n";

        //            //            if ( simd_edge_cut && i1 % 2 == 0 ) {
        //            //                code << "        __m128d s11 = _mm_set_pd1( s1 );\n";
        //            //                code << "        __m128d s02 = MM_SET_PD( s0, s2 );\n";

        //            //                code << "        __m128d m02 = _mm_div_pd( s02, _mm_sub_pd( s11, s02 ) );\n";

        //            //                code << "        __m128d x11 = _mm_set_pd1( n1.x );\n";
        //            //                code << "        __m128d y11 = _mm_set_pd1( n1.y );\n";
        //            //                code << "        __m128d x02 = MM_SET_PD( n0.x, n2.x );\n";
        //            //                code << "        __m128d y02 = MM_SET_PD( n0.y, n2.y );\n";
        //            //            } else {
        //            //                code << "        TF m0 = s0 / ( s1 - s0 );\n";
        //            //                code << "        TF m1 = s2 / ( s1 - s2 );\n";

        //            //                // save coordinates that can be modified
        //            //                code << "        TF n0_x = n0.x;\n";
        //            //                code << "        TF n0_y = n0.y;\n";
        //            //            }

        //            //            // shift points
        //            //            for( int i = size; i > in; --i )
        //            //                code << "        nodes->local_at( " << i << " ).get_straight_content_from( nodes->local_at( " << i - 1 << " ) );\n";

        //            //            // modified or added points
        //            //            if ( simd_edge_cut && i1 % 2 == 0 ) {
        //            //                code << "        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }\n";
        //            //                code << "        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }\n";

        //            //                code << "        nn.cut_id.set( n1.cut_id.get() );\n";
        //            //                code << "        n1.cut_id.set( cut_id );\n";

        //            //                code << "        __m128d x1n = _mm_sub_pd( x02, _mm_mul_pd( m02, _mm_sub_pd( x11, x02 ) ) );\n";
        //            //                code << "        __m128d y1n = _mm_sub_pd( y02, _mm_mul_pd( m02, _mm_sub_pd( y11, y02 ) ) );\n";

        //            //                if ( i1 % 2 ) {
        //            //                    code << "        n1.x = reinterpret_cast<const TF *>( &x1n )[ 0 ];\n";
        //            //                    code << "        n1.y = reinterpret_cast<const TF *>( &y1n )[ 0 ];\n";

        //            //                    code << "        nn.x = reinterpret_cast<const TF *>( &x1n )[ 1 ];\n";
        //            //                    code << "        nn.y = reinterpret_cast<const TF *>( &y1n )[ 1 ];\n";
        //            //                } else {
        //            //                    code << "        _mm_store_pd( &n1.x, x1n );\n";
        //            //                    code << "        _mm_store_pd( &n1.y, y1n );\n";
        //            //                }
        //            //            } else {
        //            //                code << "        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }\n";
        //            //                code << "        nn.x = n2.x - m1 * ( n1.x - n2.x );\n";
        //            //                code << "        nn.y = n2.y - m1 * ( n1.y - n2.y );\n";
        //            //                code << "        nn.cut_id.set( n1.cut_id.get() );\n";

        //            //                code << "        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }\n";
        //            //                code << "        n1.x = n0_x - m0 * ( n1.x - n0_x );\n";
        //            //                code << "        n1.y = n0_y - m0 * ( n1.y - n0_y );\n";
        //            //                code << "        n1.cut_id.set( cut_id );\n";
        //            //            }
        //        }
    }

    // 2 points are outside ?
    if ( nb_outside == 2 ) {
        if ( i1 == 0 && ! outside[ 1 ] )
            i1 = size - 1;

        int i0 = ( i1 + size - 1 ) % size;
        int i2 = ( i1 + 1 )        % size;
        int i3 = ( i1 + 2 )        % size;

        code << "        Node &n0 = nodes->local_at( " << i0 << " );\n";
        code << "        Node &n1 = nodes->local_at( " << i1 << " );\n";
        code << "        Node &n2 = nodes->local_at( " << i2 << " );\n";
        code << "        Node &n3 = nodes->local_at( " << i3 << " );\n";

        code << "        TF s0 = reinterpret_cast<const TF *>( &di_" << i0 / simd_size << " )[ " << i0 % simd_size << " ];\n";
        code << "        TF s1 = reinterpret_cast<const TF *>( &di_" << i1 / simd_size << " )[ " << i1 % simd_size << " ];\n";
        code << "        TF s2 = reinterpret_cast<const TF *>( &di_" << i2 / simd_size << " )[ " << i2 % simd_size << " ];\n";
        code << "        TF s3 = reinterpret_cast<const TF *>( &di_" << i3 / simd_size << " )[ " << i3 % simd_size << " ];\n";

        if ( simd_edge_cut && false ) {
            code << "        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }\n";
            code << "        n1.cut_id.set( cut_id );\n";

            code << "        __m128d s03 = MM_SET_PD( s0, s3 );\n";
            code << "        __m128d s12 = MM_SET_PD( s1, s2 );\n";

            code << "        __m128d x03 = MM_SET_PD( n0.x, n3.x );\n";
            code << "        __m128d y03 = MM_SET_PD( n0.y, n3.y );\n";

            code << "        __m128d x12 = MM_SET_PD( n1.x, n2.x );\n";
            code << "        __m128d y12 = MM_SET_PD( n1.y, n2.y );\n";

            code << "        __m128d m12 = _mm_div_pd( s03, _mm_sub_pd( s12, s03 ) );\n";

            code << "        __m128d xnn = _mm_sub_pd( x03, _mm_mul_pd( m12, _mm_sub_pd( x12, x03 ) ) );\n";
            code << "        __m128d ynn = _mm_sub_pd( y03, _mm_mul_pd( m12, _mm_sub_pd( y12, y03 ) ) );\n";

            if ( i1 % 2 || i2 != i1 + 1 ) {
                code << "        n1.x = reinterpret_cast<const TF *>( &xnn )[ 0 ];\n";
                code << "        n1.y = reinterpret_cast<const TF *>( &ynn )[ 0 ];\n";

                code << "        n2.x = reinterpret_cast<const TF *>( &xnn )[ 1 ];\n";
                code << "        n2.y = reinterpret_cast<const TF *>( &ynn )[ 1 ];\n";
            } else {
                code << "        _mm_store_pd( &n1.x, xnn );\n";
                code << "        _mm_store_pd( &n1.y, ynn );\n";
            }
        } else {
            code << "        TF m1 = s0 / ( s1 - s0 );\n";
            code << "        TF m2 = s3 / ( s2 - s3 );\n";

            code << "        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }\n";
            code << "        n1.cut_id.set( cut_id );\n";
            code << "        n1.x = n0.x - m1 * ( n1.x - n0.x );\n";
            code << "        n1.y = n0.y - m1 * ( n1.y - n0.y );\n";
            code << "        n2.x = n3.x - m2 * ( n2.x - n3.x );\n";
            code << "        n2.y = n3.y - m2 * ( n2.y - n3.y );\n";
        }

        code << "        return true;\n";
        return;
    }

    // more than 2 points are outside, outside points are before and after bit 0
    if ( i1 == 0 && outside[ size - 1 ] ) {
        int nb_inside = size - nb_outside;
        int i3 = sdot::tocnt( outside );
        i1 = nb_inside + i3;

        int i0 = i1 - 1;
        int i2 = i3 - 1;
        int in = 0;

        code << "        Node &n0 = nodes->local_at( " << i0 << " );\n";
        code << "        Node &n1 = nodes->local_at( " << i1 << " );\n";
        code << "        Node &n2 = nodes->local_at( " << i2 << " );\n";
        code << "        Node &n3 = nodes->local_at( " << i3 << " );\n";
        code << "        Node &nn = nodes->local_at( " << in << " );\n";

        code << "        TF s0 = reinterpret_cast<const TF *>( &di_" << i0 / simd_size << " )[ " << i0 % simd_size << " ];\n";
        code << "        TF s1 = reinterpret_cast<const TF *>( &di_" << i1 / simd_size << " )[ " << i1 % simd_size << " ];\n";
        code << "        TF s2 = reinterpret_cast<const TF *>( &di_" << i2 / simd_size << " )[ " << i2 % simd_size << " ];\n";
        code << "        TF s3 = reinterpret_cast<const TF *>( &di_" << i3 / simd_size << " )[ " << i3 % simd_size << " ];\n";

        code << "        TF m1 = s0 / ( s1 - s0 );\n";
        code << "        TF m2 = s3 / ( s2 - s3 );\n";

        // modified and shifted points
        code << "        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }\n";
        code << "        nn.x = n3.x - m2 * ( n2.x - n3.x );\n";
        code << "        nn.y = n3.y - m2 * ( n2.y - n3.y );\n";
        code << "        nn.cut_id.set( n2.cut_id.get() );\n";

        int o = 1;
        for( ; o <= nb_inside; ++o )
            code << "        nodes->local_at( " << o << " ).get_straight_content_from( nodes->local_at( " << i2 + o << " ) );\n";
        code << "        Node &no = nodes->local_at( " << o << " );\n";

        code << "        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }\n";
        code << "        no.x = n0.x - m1 * ( n1.x - n0.x );\n";
        code << "        no.y = n0.y - m1 * ( n1.y - n0.y );\n";
        code << "        no.cut_id.set( cut_id );\n";

        code << "        size = " << size - ( nb_outside - 2 ) << ";\n";
        code << "        return true;\n";
        return;
    }

    // more than 2 points are outside, outside points do not cross `nb_points`
    int i0 = ( i1 + size - 1       ) % size;
    int i2 = ( i1 + nb_outside - 1 ) % size;
    int i3 = ( i1 + nb_outside     ) % size;
    int in = i1 + 1;

    code << "        Node &n0 = nodes->local_at( " << i0 << " );\n";
    code << "        Node &n1 = nodes->local_at( " << i1 << " );\n";
    code << "        Node &n2 = nodes->local_at( " << i2 << " );\n";
    code << "        Node &n3 = nodes->local_at( " << i3 << " );\n";
    code << "        Node &nn = nodes->local_at( " << in << " );\n";

    code << "        TF s0 = reinterpret_cast<const TF *>( &di_" << i0 / simd_size << " )[ " << i0 % simd_size << " ];\n";
    code << "        TF s1 = reinterpret_cast<const TF *>( &di_" << i1 / simd_size << " )[ " << i1 % simd_size << " ];\n";
    code << "        TF s2 = reinterpret_cast<const TF *>( &di_" << i2 / simd_size << " )[ " << i2 % simd_size << " ];\n";
    code << "        TF s3 = reinterpret_cast<const TF *>( &di_" << i3 / simd_size << " )[ " << i3 % simd_size << " ];\n";

    code << "        TF m1 = s0 / ( s1 - s0 );\n";
    code << "        TF m2 = s3 / ( s2 - s3 );\n";

    // modified and deleted points
    code << "        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }\n";
    code << "        n1.x = n0.x - m1 * ( n1.x - n0.x );\n";
    code << "        n1.y = n0.y - m1 * ( n1.y - n0.y );\n";
    code << "        n1.cut_id.set( cut_id );\n";

    code << "        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }\n";
    code << "        nn.x = n3.x - m2 * ( n2.x - n3.x );\n";
    code << "        nn.y = n3.y - m2 * ( n2.y - n3.y );\n";
    code << "        nn.cut_id.set( n2.cut_id.get() );\n";

    std::size_t nb_to_rem = nb_outside - 2;
    for( int i = i2 + 1; i < size; ++i )
        code << "        nodes->local_at( " << i - nb_to_rem << " ).get_straight_content_from( nodes->local_at( " << i << " ) );\n";

    // modification of the number of points
    code << "        size = " << size - nb_to_rem << ";\n";
    code << "        return true;\n";
}

void generate( int simd_size, std::string /*ext*/, int max_size_included = 8 ) {
    int mul_size = 1 << max_size_included;
    int max_index = mul_size * ( max_size_included + 1 );
    int nb_regs = ( max_size_included + simd_size - 1 ) / simd_size;

    std::cout << "    // outsize list\n";
    std::cout << "    __m512d ox = _mm512_set1_pd( origin.x );\n";
    std::cout << "    __m512d oy = _mm512_set1_pd( origin.y );\n";
    std::cout << "    __m512d nx = _mm512_set1_pd( normal.x );\n";
    std::cout << "    __m512d ny = _mm512_set1_pd( normal.y );\n";
    for( int i = 0; i < nb_regs; ++i ) {
        std::cout << "    __m512d px_" << i << " = _mm512_load_pd( &nodes->x + " << simd_size * i << " );\n";
        std::cout << "    __m512d py_" << i << " = _mm512_load_pd( &nodes->y + " << simd_size * i << " );\n";
        std::cout << "    __m512d di_" << i << " = _mm512_add_pd( _mm512_mul_pd( _mm512_sub_pd( ox, px_" << i << " ), nx ), _mm512_mul_pd( _mm512_sub_pd( oy, py_" << i << " ), ny ) );\n";
        std::cout << "    std::uint8_t outside_" << i << " = _mm512_cmp_pd_mask( di_" << i << ", _mm512_set1_pd( 0.0 ), _CMP_LT_OQ ); // OS?\n"; // OQ => 46.9, QS => 47.1
        // std::cout << "    std::uint8_t outside_" << i << " = _mm512_movepi64_mask( __m512i( di_" << i << " ) );\n"; // => 47.1
    }
    std::cout << "    \n";

    // gather
    std::map<std::string,std::vector<int>> cases;
    for( int index = 0; index < max_index; ++index ) {
        std::ostringstream code;
        get_code( code, index, max_size_included, simd_size );
        cases[ code.str() ].push_back( index );
    }

    // write
    std::cout << "    switch( " << mul_size << " * size + ";
    for( int i = 0; i < nb_regs; ++i )
        std::cout << ( 1 << ( i * simd_size ) ) <<  " * outside_" << i;
    std::cout << " ) {";
    for( std::pair<std::string,std::vector<int>> c : cases ) {
        for( int index : c.second )
            std::cout << "\n    case " << index << ":";
        std::cout << " {\n" << c.first << "    }";
    }
    std::cout << "    default: break;\n";
    std::cout << "    }\n";
}

int main() {
    std::cout << "#include \"../ConvexPolyhedron2.h\"\n";
    std::cout << "#define MM_SET_PD( A, B ) _mm_set_pd( B, A ) \n";
    std::cout << "#define MM256_SET_PD( A, B, C, D ) _mm256_set_pd( D, C, B, A ) \n";
    std::cout << "namespace sdot {\n";

    //
    std::cout << "\n";
    std::cout << "template<class Pc> template<int flags>\n";
    std::cout << "bool ConvexPolyhedron2<Pc>::plane_cut_simd_switch( Pt origin, Pt normal, CI cut_id, N<flags>, S<double> ) {\n";
    std::cout << "    #ifdef __AVX512F__\n";
    generate( 8, "AVX512", 8 );
    std::cout << "    #endif // __AVX512F__\n";
    std::cout << "    return plane_cut_gen( origin, normal, cut_id, N<flags>() );\n";
    std::cout << "}\n";

    // generic version
    std::cout << "\n";
    std::cout << "template<class Pc> template<int flags,class T>\n";
    std::cout << "bool ConvexPolyhedron2<Pc>::plane_cut_simd_switch( Pt origin, Pt normal, CI cut_id, N<flags>, S<T> ) {\n";
    std::cout << "    return plane_cut_gen( origin, normal, cut_id, N<flags>() );\n";
    std::cout << "}\n";

    std::cout << "\n";
    std::cout << "} // namespace sdot\n";
}


