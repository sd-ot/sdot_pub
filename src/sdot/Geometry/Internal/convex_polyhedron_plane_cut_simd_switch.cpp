// #include <immintrin.h>
#include "../../Support/bit_handling.h"
#include <iostream>
#include <sstream>
#include <vector>
#include <map>



void get_code( std::ostringstream &code, int index, int max_size_included, int simd_size ) {
    int mul_size = 1 << max_size_included;

    int size = index / mul_size;
    std::bitset<64> outside = index & ( ( 1 << size ) - 1 );
    int nb_outside = sdot::popcnt( outside );
    if ( nb_outside == 0 ) {
        code << "        return false;\n";
        return;
    }
    if ( nb_outside == size ) {
        code << "        size = 0;\n";
        code << "        return true;\n";
        return;
    }

    // update normal
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

        code << "        Node &n0 = node( " << i0 << " );\n";
        code << "        Node &n1 = node( " << i1 << " );\n";
        code << "        Node &n2 = node( " << i2 << " );\n";
        code << "        Node &nn = node( " << in << " );\n";

        code << "        TF s0 = reinterpret_cast<const TF *>( &di_" << i0 / simd_size << " )[ " << i0 % simd_size << " ];\n";
        code << "        TF s1 = reinterpret_cast<const TF *>( &di_" << i1 / simd_size << " )[ " << i1 % simd_size << " ];\n";
        code << "        TF s2 = reinterpret_cast<const TF *>( &di_" << i2 / simd_size << " )[ " << i2 % simd_size << " ];\n";

        code << "        TF m0 = s0 / ( s1 - s0 );\n";
        code << "        TF m1 = s2 / ( s1 - s2 );\n";

        // save coordinates that can be modified
        code << "        TF n0_x = n0.x;\n";
        code << "        TF n0_y = n0.y;\n";

        // shift points
        for( int i = size; i > in; --i )
            code << "        node( " << i << " ).get_straight_content_from( node( " << i - 1 << " ) );\n";

        // modified or added points
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
    }

    // 2 points are outside ?
    if ( nb_outside == 2 ) {
        if ( i1 == 0 && ! outside[ 1 ] )
            i1 = size - 1;

        int i0 = ( i1 + size - 1 ) % size;
        int i2 = ( i1 + 1 )        % size;
        int i3 = ( i1 + 2 )        % size;

        code << "        Node &n0 = node( " << i0 << " );\n";
        code << "        Node &n1 = node( " << i1 << " );\n";
        code << "        Node &n2 = node( " << i2 << " );\n";
        code << "        Node &n3 = node( " << i3 << " );\n";

        code << "        TF s0 = reinterpret_cast<const TF *>( &di_" << i0 / simd_size << " )[ " << i0 % simd_size << " ];\n";
        code << "        TF s1 = reinterpret_cast<const TF *>( &di_" << i1 / simd_size << " )[ " << i1 % simd_size << " ];\n";
        code << "        TF s2 = reinterpret_cast<const TF *>( &di_" << i2 / simd_size << " )[ " << i2 % simd_size << " ];\n";
        code << "        TF s3 = reinterpret_cast<const TF *>( &di_" << i3 / simd_size << " )[ " << i3 % simd_size << " ];\n";

        code << "        TF m1 = s0 / ( s1 - s0 );\n";
        code << "        TF m2 = s3 / ( s2 - s3 );\n";

        // modified points
        code << "        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }\n";
        code << "        n1.cut_id.set( cut_id );\n";
        code << "        n1.x = n0.x - m1 * ( n1.x - n0.x );\n";
        code << "        n1.y = n0.y - m1 * ( n1.y - n0.y );\n";
        code << "        n2.x = n3.x - m2 * ( n2.x - n3.x );\n";
        code << "        n2.y = n3.y - m2 * ( n2.y - n3.y );\n";

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

        code << "        Node &n0 = node( " << i0 << " );\n";
        code << "        Node &n1 = node( " << i1 << " );\n";
        code << "        Node &n2 = node( " << i2 << " );\n";
        code << "        Node &n3 = node( " << i3 << " );\n";
        code << "        Node &nn = node( " << in << " );\n";

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
        code << "        Node &no = node( " << o << " );\n";

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

    code << "        Node &n0 = node( " << i0 << " );\n";
    code << "        Node &n1 = node( " << i1 << " );\n";
    code << "        Node &n2 = node( " << i2 << " );\n";
    code << "        Node &n3 = node( " << i3 << " );\n";
    code << "        Node &nn = node( " << in << " );\n";

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
        std::cout << "    std::uint8_t outside_" << i << " = _mm512_cmp_pd_mask( di_" << i << ", _mm512_set1_pd( 0.0 ), _CMP_LT_OQ ); // OS?\n";
        // std::cout << "    std::uint8_t outside = _mm512_movepi64_mask( di_" << i << " );\n";
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
    //    template<class Pc> template<int flags>
    //    bool ConvexPolyhedron2<Pc>::plane_cut_simd_switch( Pt origin, Pt normal, CI cut_id, N<flags> ) {
    //        #include "Internal/(convex_polyhedron_plane_cut_simd_switch.cpp).h"
    //    }
    std::cout << "#include \"../ConvexPolyhedron2.h\"\n";
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



//constexpr std::size_t mul_size = 64;
//switch ( mul_size * size + outside ) {
//case mul_size * 4 + 0b0000:
//    return false;
//case mul_size * 4 + 0b0001:
//    TODO;
//    return true;
//case mul_size * 4 + 0b0010:
//    TODO;
//    return true;
//case mul_size * 4 + 0b0011:
//    TODO;
//    return true;
//case mul_size * 4 + 0b0100:
//    TODO;
//    return true;
//case mul_size * 4 + 0b0101:
//    TODO;
//    return true;
//case mul_size * 4 + 0b0110: {
//    TF s0 = d[ 0 ];
//    TF s1 = d[ 1 ];
//    TF s2 = d[ 2 ];
//    TF s3 = d[ 3 ];

//    TF m1 = s0 / ( s1 - s0 );
//    TF m2 = s3 / ( s2 - s3 );

//    Node &n0 = nodes->local_at( 0 );
//    Node &n1 = nodes->local_at( 1 );
//    Node &n2 = nodes->local_at( 2 );
//    Node &n3 = nodes->local_at( 3 );

//    if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
//    n1.x = n0.x - m1 * ( n1.x - n0.x );
//    n1.y = n0.y - m1 * ( n1.y - n0.y );
//    n2.x = n3.x - m2 * ( n2.x - n3.x );
//    n2.y = n3.y - m2 * ( n2.y - n3.y );
//    n1.cut_id.set( cut_id );
//    return true;
//}
//case mul_size * 4 + 0b0111:
//    TODO;
//    return true;
//case mul_size * 4 + 0b1000: {
//    // no point to shift
//    //        for( std::size_t i = size; i > i1 + 1; --i ) {
//    //            points [ 0 ][ i ] = points [ 0 ][ i - 1 ];
//    //            points [ 1 ][ i ] = points [ 1 ][ i - 1 ];
//    //            cut_ids     [ i ] = cut_ids     [ i - 1 ];
//    //            if ( store_the_normals ) {
//    //                normals[ 0 ][ i ] = normals[ 0 ][ i - 1 ];
//    //                normals[ 1 ][ i ] = normals[ 1 ][ i - 1 ];
//    //            }
//    //        }

//    constexpr std::size_t si = 4;
//    constexpr int i1 = 3;
//    constexpr int i0 = ( i1 + si - 1 ) % si;
//    constexpr int i2 = ( i1 + 1 ) % si;
//    constexpr int in = 4;

//    xsimd::batch<TF,2> s1{ d[ i1 ] };
//    xsimd::batch<TF,2> so{ d[ i0 ], d[ i2 ] };
//    xsimd::batch<TF,2> mo = so / ( s1 - so );

//    size = 5;

//    Node &n0 = nodes->local_at( i0 );
//    Node &n1 = nodes->local_at( i1 );
//    Node &n2 = nodes->local_at( i2 );
//    Node &nn = nodes->local_at( in );

//    if ( store_the_normals ) {
//        nn.dir_x = n1.dir_x;
//        nn.dir_y = n1.dir_y;
//        n1.dir_x = normal.x;
//        n1.dir_y = normal.y;
//    }

//    //
//    xsimd::batch<TF,2> x1{ n1.x };
//    xsimd::batch<TF,2> y1{ n1.y };

//    xsimd::batch<TF,2> xo{ n0.x, n2.x };
//    xsimd::batch<TF,2> yo{ n0.y, n2.y };

//    xsimd::batch<TF,2> nx = xo - mo * ( x1 - xo );
//    xsimd::batch<TF,2> ny = yo - mo * ( y1 - xo );

//    nx.store_unaligned( &n1.x );
//    ny.store_unaligned( &n1.y );

//    nn.cut_id.set( n1.cut_id.get() );
//    n1.cut_id.set( cut_id );
//    return true;
//}
//case mul_size * 4 + 0b1001:
//    TODO;
//    return true;
//case mul_size * 4 + 0b1010:
//    TODO;
//    return true;
//case mul_size * 4 + 0b1011:
//    TODO;
//    return true;
//case mul_size * 4 + 0b1100:
//    TODO;
//    return true;
//case mul_size * 4 + 0b1101:
//    TODO;
//    return true;
//case mul_size * 4 + 0b1110: {
//    constexpr std::size_t si = 4;
//    constexpr std::size_t i1 = 1;
//    constexpr std::size_t nb_outside = 3;
//    constexpr std::size_t i0 = ( i1 + si  - 1        ) % si;
//    constexpr std::size_t i2 = ( i1 + nb_outside - 1 ) % si;
//    constexpr std::size_t i3 = ( i1 + nb_outside     ) % si;
//    constexpr std::size_t in = i1 + 1;

//    // more than 2 points are outside, outside points are before and after bit 0
//    // if ( i1 == 0 && ( outside & ( 1ul << ( size - 1 ) ) ) ) {
//    //

//    Node &n0 = nodes->local_at( i0 );
//    Node &n1 = nodes->local_at( i1 );
//    Node &n2 = nodes->local_at( i2 );
//    Node &n3 = nodes->local_at( i3 );
//    Node &nn = nodes->local_at( in );

//    TF s0 = d[ i0 ];
//    TF s1 = d[ i1 ];
//    TF s2 = d[ i2 ];
//    TF s3 = d[ i3 ];

//    TF m1 = s0 / ( s1 - s0 );
//    TF m2 = s3 / ( s2 - s3 );

//    // modified and deleted points
//    n1.x = n0.x - m1 * ( n1.x - n0.x );
//    n1.y = n0.y - m1 * ( n1.y - n0.y );
//    n1.cut_id.set( cut_id );
//    if ( store_the_normals ) {
//        n1.dir_x = normal.x;
//        n1.dir_y = normal.y;
//    }

//    nn.x = n3.x - m2 * ( n2.x - n3.x );
//    nn.y = n3.y - m2 * ( n2.y - n3.y );
//    nn.cut_id.set( n2.cut_id.get() );
//    if ( store_the_normals ) {
//        nn.dir_x = n2.dir_x;
//        nn.dir_y = n2.dir_y;
//    }

//    constexpr std::size_t nb_to_rem = nb_outside - 2;
//    for( std::size_t i = i2 + 1; i < si; ++i )
//        nodes->local_at( i - nb_to_rem ).get_straight_content_from( nodes->local_at( i ) );

//    // modification of the number of points
//    size -= nb_to_rem;
//    return true;
//}
//case mul_size * 4 + 0b1111:
//    size = 0;
//    return true;

//// ------------------------------------------------
//case mul_size * 3 + 0b000:
//    return false;
//case mul_size * 3 + 0b001:
//    TODO;
//    return true;
//case mul_size * 3 + 0b010:
//    TODO;
//    return true;
//case mul_size * 3 + 0b011:
//    TODO;
//    return true;
//case mul_size * 3 + 0b100:
//    TODO;
//    return true;
//case mul_size * 3 + 0b101:
//    TODO;
//    return true;
//case mul_size * 3 + 0b110: {

//    return true;
//}
//case mul_size * 3 + 0b111:
//    TODO;
//    return true;
//default:
//    return false;
//}
