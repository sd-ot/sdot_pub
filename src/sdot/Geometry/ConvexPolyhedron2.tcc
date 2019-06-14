#include "../Support/Display/binary_repr.h"
#include "../Support/aligned_memory.h"
#include "../Support/xsimd_util.h"
#include "../Support/bit_handling.h"
#include "../Support/TODO.h"
#include "../Support/P.h"
#include "ConvexPolyhedron2.h"
#include <xsimd/xsimd.hpp>
#include <iomanip>
#include <bitset>

//#define _USE_MATH_DEFINES
//#include <math.h>

namespace sdot {

template<class Pc>
ConvexPolyhedron2<Pc>::ConvexPolyhedron2( const Box &box, CI cut_id ) : ConvexPolyhedron2() {
    resize( 4 );

    // points
    Node &n0 = node( 0 );
    Node &n1 = node( 1 );
    Node &n2 = node( 2 );
    Node &n3 = node( 3 );

    n0.x = box.p0[ 0 ]; n0.y = box.p0[ 1 ];
    n1.x = box.p1[ 0 ]; n1.y = box.p0[ 1 ];
    n2.x = box.p1[ 0 ]; n2.y = box.p1[ 1 ];
    n3.x = box.p0[ 0 ]; n3.y = box.p1[ 1 ];

    // normals
    n0.dir_x =  0; n0.dir_y = -1;
    n1.dir_x = +1; n1.dir_y =  0;
    n2.dir_x =  0; n2.dir_y = +1;
    n3.dir_x = -1; n3.dir_y =  0;

    // cut_ids
    n0.cut_id.set( cut_id );
    n1.cut_id.set( cut_id );
    n2.cut_id.set( cut_id );
    n3.cut_id.set( cut_id );
}

template<class Pc>
ConvexPolyhedron2<Pc>::ConvexPolyhedron2() {
    size = 0;
    rese = block_size;
    nodes = new ( aligned_malloc( rese / block_size * sizeof( Node ), 32 ) ) Node;
}

template<class Pc>
ConvexPolyhedron2<Pc>::~ConvexPolyhedron2() {
    if ( rese )
        aligned_free( nodes );
}

template<class Pc>
ConvexPolyhedron2<Pc> &ConvexPolyhedron2<Pc>::operator=( const ConvexPolyhedron2 &that ) {
    resize( that.nb_nodes() );
    for( std::size_t i = 0; ; i += block_size ) {
        Node &n = nodes[ i / block_size ];
        const Node &t = that.nodes[ i / block_size ];
        if ( i + block_size > size ) {
            std::size_t r = size - i;
            memcpy( &n.x, &t.x, r * sizeof( TF ) );
            memcpy( &n.y, &t.y, r * sizeof( TF ) );
            if ( store_the_normals ) {
                memcpy( &n.dir_x, &t.dir_x, r * sizeof( TF ) );
                memcpy( &n.dir_y, &t.dir_y, r * sizeof( TF ) );
            }
            if ( allow_ball_cut ) {
                memcpy( &n.arc_radius  , &t.arc_radius  , r * sizeof( TF ) );
                memcpy( &n.arc_center_x, &t.arc_center_x, r * sizeof( TF ) );
                memcpy( &n.arc_center_y, &t.arc_center_y, r * sizeof( TF ) );
            }
            for( std::size_t j = 0; j < r; ++j )
                n.local_at( j ).cut_id.set( t.local_at( j ).cut_id.get() );
            break;
        }

        memcpy( &n.x, &t.x, block_size * sizeof( TF ) );
        memcpy( &n.y, &t.y, block_size * sizeof( TF ) );
        if ( store_the_normals ) {
            memcpy( &n.dir_x, &t.dir_x, block_size * sizeof( TF ) );
            memcpy( &n.dir_y, &t.dir_y, block_size * sizeof( TF ) );
        }
        if ( allow_ball_cut ) {
            memcpy( &n.arc_radius  , &t.arc_radius  , block_size * sizeof( TF ) );
            memcpy( &n.arc_center_x, &t.arc_center_x, block_size * sizeof( TF ) );
            memcpy( &n.arc_center_y, &t.arc_center_y, block_size * sizeof( TF ) );
        }
        for( std::size_t j = 0; j < block_size; ++j )
            n.local_at( j ).cut_id.set( t.local_at( j ).cut_id.get() );
    }


    return *this;
}

template<class Pc>
void ConvexPolyhedron2<Pc>::write_to_stream( std::ostream &os ) const {
    os << "pos: ";
    for( TI i = 0; i < nb_nodes(); ++i )
        os << ( i ? " [" : "[" ) << node( i ).pos() << "]";
    if ( store_the_normals ) {
        os << " nrms: ";
        for( TI i = 0; i < nb_nodes(); ++i )
            os << ( i ? " [" : "[" ) << node( i ).dir() << "]";
    }
    //    os << " sphere center: " << sphere_center << " sphere radius: " << sphere_radius;
}

template<class Pc>
void ConvexPolyhedron2<Pc>::display( VtkOutput &vo, const std::vector<TF> &cell_values, Pt offset ) const {
    std::vector<VtkOutput::Pt> pts;
    pts.reserve( nb_nodes() );
    for_each_node( [&]( const Node &node ) {
        pts.push_back( node.pos() + offset );
    } );
    vo.add_polygon( pts, cell_values );
}

template<class Pc> template<class F>
void ConvexPolyhedron2<Pc>::for_each_edge( const F &/*f*/ ) const {
    TODO;
}

template<class Pc> template<class F>
void ConvexPolyhedron2<Pc>::for_each_node( const F &f ) const {
    static_assert ( sizeof( Node ) % sizeof( TF ) == 0, "" );

    Node *ptr = nodes;
    for( TI i = 0; ; ++i ) {
        if ( i + block_size >= size ) {
            for( TI j = 0; j < size - i; ++j )
                f( ptr->local_at( j ) );
            break;
        }

        for( TI j = 0; j < block_size; ++j )
            f( ptr->local_at( j ) );
        i += block_size;
        ++ptr;
    }
}

template<class Pc>
typename ConvexPolyhedron2<Pc>::TI ConvexPolyhedron2<Pc>::nb_nodes() const {
    return size;
}

template<class Pc>
const typename ConvexPolyhedron2<Pc>::Node &ConvexPolyhedron2<Pc>::node( TI index ) const {
    return nodes->global_at( index );
}

template<class Pc>
typename ConvexPolyhedron2<Pc>::Node &ConvexPolyhedron2<Pc>::node( TI index ) {
    return nodes->global_at( index );
}

template<class Pc>
void ConvexPolyhedron2<Pc>::resize( TI new_size ) {
    if ( new_size > rese ) {
        TI old_nb_blocks = rese / block_size;
        Node *old_nodes = nodes;

        TI nb_blocks = ( new_size + block_size - 1 ) / block_size;
        rese = nb_blocks * block_size;

        nodes = reinterpret_cast<Node *>( aligned_malloc( nb_blocks * sizeof( Node ), 32 ) );
        for( TI i = 0; i < old_nb_blocks; ++i )
            new ( nodes + i ) Node( std::move( old_nodes[ i ] ) );

        if ( old_nb_blocks )
            aligned_free( old_nodes );
    }
    size = new_size;
}

template<class Pc>
bool ConvexPolyhedron2<Pc>::plane_cut( Pt origin, Pt normal, CI cut_id ) {
    return plane_cut( origin, normal, cut_id, N<0>() );
}

template<class Pc> template<int flags>
bool ConvexPolyhedron2<Pc>::plane_cut_simd_tzcnt( Pt origin, Pt normal, CI cut_id, N<flags> ) {
    constexpr std::size_t simd_size = xsimd::simd_type<TF>::size;
    using BB = xsimd::batch_bool<TF,simd_size>;
    using BF = xsimd::batch<TF,simd_size>;

    // outsize list
    BF ox( origin.x );
    BF oy( origin.y );
    BF nx( normal.x );
    BF ny( normal.y );

    BF px; px.load_aligned( &nodes->x );
    BF py; py.load_aligned( &nodes->y );
    BF d = ( ox - px ) * nx + ( oy - py ) * ny;
    std::uint64_t outside = is_neg( d );

    // all inside ?
    if ( outside == 0 )
        return false;

    // all outside ?
    unsigned nb_outside = popcnt( outside );
    if ( nb_outside == size ) {
        size = 0;
        return false;
    }

    // => we will need a normalized direction
    if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 ) {
        TF n = 1 / norm_2( normal );
        for( std::size_t i = 0; i < size / simd_size; ++i )
            d[ i ] *= n;
        normal = n * normal;
    }

    // => we will need index of the first outside point
    const TF *distances = reinterpret_cast<const TF *>( &d );
    std::size_t i1 = tzcnt( outside );

    // only 1 outside ?
    if ( nb_outside == 1 ) {
        // => creation of a new point
        std::size_t i0 = ( i1 + size - 1 ) % size;
        std::size_t i2 = ( i1 + 1 ) % size;
        std::size_t in = i1 + 1;

        Node &n0 = node( i0 );
        Node &n1 = node( i1 );
        Node &n2 = node( i2 );
        Node &nn = node( in );

        TF s0 = distances[ i0 ];
        TF s1 = distances[ i1 ];
        TF s2 = distances[ i2 ];

        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );

        // shift points
        for( std::size_t i = size; i > i1 + 1; --i )
            node( i + 1 ).get_straight_content_from( node( i + 0 ) );
        ++size;

        // modified or added points
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) {
            nn.dir_x = n1.dir_x;
            nn.dir_y = n1.dir_y;
        }

        n1.x = n0.x - m0 * ( n1.x - n0.x );
        n1.y = n0.y - m0 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) {
            n1.dir_x = normal.x;
            n1.dir_y = normal.y;
        }

        return true;
    }

    // 2 points are outside ?
    if ( nb_outside == 2 ) {
        if ( i1 == 0 && ! ( outside & 2 ) )
            i1 = size - 1;

        std::size_t i0 = ( i1 + size - 1 ) % size;
        std::size_t i2 = ( i1 + 1 )        % size;
        std::size_t i3 = ( i1 + 2 )        % size;

        Node &n0 = node( i0 );
        Node &n1 = node( i1 );
        Node &n2 = node( i2 );
        Node &n3 = node( i3 );

        TF s0 = distances[ i0 ];
        TF s1 = distances[ i1 ];
        TF s2 = distances[ i2 ];
        TF s3 = distances[ i3 ];

        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );

        // modified points
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        nodes->local_at( 1 ).cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }

    // more than 2 points are outside, outside points are before and after bit 0
    if ( i1 == 0 && ( outside & ( 1ul << ( size - 1 ) ) ) ) {
        std::size_t nb_inside = size - nb_outside;
        std::size_t i3 = tzcnt( ~ outside );
        i1 = nb_inside + i3;
        std::size_t i0 = i1 - 1;
        std::size_t i2 = i3 - 1;

        Node &n0 = node( i0 );
        Node &n1 = node( i1 );
        Node &n2 = node( i2 );
        Node &n3 = node( i3 );
        Node &nz = node(  0 );

        Pt p0 { n0.x, n0.y };
        Pt p1 { n1.x, n1.y };
        Pt p2 { n2.x, n2.y };
        Pt p3 { n3.x, n3.y };

        TF s0 = distances[ i0 ];
        TF s1 = distances[ i1 ];
        TF s2 = distances[ i2 ];
        TF s3 = distances[ i3 ];

        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );

        // modified and shifted points
        nz.x = p3.x - m2 * ( p2.x - p3.x );
        nz.y = p3.y - m2 * ( p2.y - p3.y );
        nz.cut_id.set( n2.cut_id.get() );
        if ( store_the_normals ) {
            nz.x = n2.dir_x;
            nz.y = n2.dir_y;
        }
        std::size_t o = 1;
        for( ; o <= nb_inside; ++o )
            node( o ).get_straight_content_from( node( i2 + o ) );
        Node &no = node( o );
        no.x = p0.x - m1 * ( p1.x - p0.x );
        no.y = p0.y - m1 * ( p1.y - p0.y );
        no.cut_id.set( cut_id );
        if ( store_the_normals ) {
            no.dir_x = normal.x;
            no.dir_y = normal.y;
        }

        size -= nb_outside - 2;
        return true;
    }

    // more than 2 points are outside, outside points do not cross `nb_points`
    std::size_t i0 = ( i1 + size - 1       ) % size;
    std::size_t i2 = ( i1 + nb_outside - 1 ) % size;
    std::size_t i3 = ( i1 + nb_outside     ) % size;
    std::size_t in = i1 + 1;

    Node &n0 = node( i0 );
    Node &n1 = node( i1 );
    Node &n2 = node( i2 );
    Node &n3 = node( i3 );
    Node &nn = node( in );

    TF s0 = distances[ i0 ];
    TF s1 = distances[ i1 ];
    TF s2 = distances[ i2 ];
    TF s3 = distances[ i3 ];

    TF m1 = s0 / ( s1 - s0 );
    TF m2 = s3 / ( s2 - s3 );

    // modified and deleted points
    if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
    n1.x = n0.x - m1 * ( n1.x - n0.x );
    n1.y = n0.y - m1 * ( n1.y - n0.y );
    n1.cut_id.set( cut_id );

    if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
    nn.x = n3.x - m2 * ( n2.x - n3.x );
    nn.y = n3.y - m2 * ( n2.y - n3.y );
    nn.cut_id.set( n2.cut_id.get() );

    std::size_t nb_to_rem = nb_outside - 2;
    for( std::size_t i = i2 + 1; i < size; ++i )
        nodes->local_at( i - nb_to_rem ).get_straight_content_from( nodes->local_at( i ) );

    // modification of the number of points
    size -= nb_to_rem;
    return true;
}

template<class Pc> template<int flags>
bool ConvexPolyhedron2<Pc>::plane_cut_simd_switch( Pt origin, Pt normal, CI cut_id, N<flags> ) {
    constexpr std::size_t simd_size = xsimd::simd_type<TF>::size;
    using BB = xsimd::batch_bool<TF,simd_size>;
    using BF = xsimd::batch<TF,simd_size>;

    // outsize list
    BF ox( origin.x );
    BF oy( origin.y );
    BF nx( normal.x );
    BF ny( normal.y );

    BF px; px.load_aligned( &nodes->x );
    BF py; py.load_aligned( &nodes->y );
    BF d = ( ox - px ) * nx + ( oy - py ) * ny;
    std::uint64_t outside = is_neg( d );

    constexpr std::size_t mul_size = 64;
    switch ( mul_size * size + outside ) {
    case mul_size * 4 + 0b0000:
        return false;
    case mul_size * 4 + 0b0001:
        TODO;
        return true;
    case mul_size * 4 + 0b0010:
        TODO;
        return true;
    case mul_size * 4 + 0b0011:
        TODO;
        return true;
    case mul_size * 4 + 0b0100:
        TODO;
        return true;
    case mul_size * 4 + 0b0101:
        TODO;
        return true;
    case mul_size * 4 + 0b0110: {
        TF s0 = d[ 0 ];
        TF s1 = d[ 1 ];
        TF s2 = d[ 2 ];
        TF s3 = d[ 3 ];

        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );

        Node &n0 = nodes->local_at( 0 );
        Node &n1 = nodes->local_at( 1 );
        Node &n2 = nodes->local_at( 2 );
        Node &n3 = nodes->local_at( 3 );

        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case mul_size * 4 + 0b0111:
        TODO;
        return true;
    case mul_size * 4 + 0b1000: {
        // no point to shift
        //        for( std::size_t i = size; i > i1 + 1; --i ) {
        //            points [ 0 ][ i ] = points [ 0 ][ i - 1 ];
        //            points [ 1 ][ i ] = points [ 1 ][ i - 1 ];
        //            cut_ids     [ i ] = cut_ids     [ i - 1 ];
        //            if ( store_the_normals ) {
        //                normals[ 0 ][ i ] = normals[ 0 ][ i - 1 ];
        //                normals[ 1 ][ i ] = normals[ 1 ][ i - 1 ];
        //            }
        //        }

        constexpr std::size_t si = 4;
        constexpr int i1 = 3;
        constexpr int i0 = ( i1 + si - 1 ) % si;
        constexpr int i2 = ( i1 + 1 ) % si;
        constexpr int in = 4;

        xsimd::batch<TF,2> s1{ d[ i1 ] };
        xsimd::batch<TF,2> so{ d[ i0 ], d[ i2 ] };
        xsimd::batch<TF,2> mo = so / ( s1 - so );

        size = 5;

        Node &n0 = nodes->local_at( i0 );
        Node &n1 = nodes->local_at( i1 );
        Node &n2 = nodes->local_at( i2 );
        Node &nn = nodes->local_at( in );

        if ( store_the_normals ) {
            nn.dir_x = n1.dir_x;
            nn.dir_y = n1.dir_y;
            n1.dir_x = normal.x;
            n1.dir_y = normal.y;
        }

        //
        xsimd::batch<TF,2> x1{ n1.x };
        xsimd::batch<TF,2> y1{ n1.y };

        xsimd::batch<TF,2> xo{ n0.x, n2.x };
        xsimd::batch<TF,2> yo{ n0.y, n2.y };

        xsimd::batch<TF,2> nx = xo - mo * ( x1 - xo );
        xsimd::batch<TF,2> ny = yo - mo * ( y1 - xo );

        nx.store_unaligned( &n1.x );
        ny.store_unaligned( &n1.y );

        nn.cut_id.set( n1.cut_id.get() );
        n1.cut_id.set( cut_id );
        return true;
    }
    case mul_size * 4 + 0b1001:
        TODO;
        return true;
    case mul_size * 4 + 0b1010:
        TODO;
        return true;
    case mul_size * 4 + 0b1011:
        TODO;
        return true;
    case mul_size * 4 + 0b1100:
        TODO;
        return true;
    case mul_size * 4 + 0b1101:
        TODO;
        return true;
    case mul_size * 4 + 0b1110: {
        constexpr std::size_t si = 4;
        constexpr std::size_t i1 = 1;
        constexpr std::size_t nb_outside = 3;
        constexpr std::size_t i0 = ( i1 + si  - 1        ) % si;
        constexpr std::size_t i2 = ( i1 + nb_outside - 1 ) % si;
        constexpr std::size_t i3 = ( i1 + nb_outside     ) % si;
        constexpr std::size_t in = i1 + 1;

        // more than 2 points are outside, outside points are before and after bit 0
        // if ( i1 == 0 && ( outside & ( 1ul << ( size - 1 ) ) ) ) {
        //

        Node &n0 = nodes->local_at( i0 );
        Node &n1 = nodes->local_at( i1 );
        Node &n2 = nodes->local_at( i2 );
        Node &n3 = nodes->local_at( i3 );
        Node &nn = nodes->local_at( in );

        TF s0 = d[ i0 ];
        TF s1 = d[ i1 ];
        TF s2 = d[ i2 ];
        TF s3 = d[ i3 ];

        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );

        // modified and deleted points
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) {
            n1.dir_x = normal.x;
            n1.dir_y = normal.y;
        }

        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        if ( store_the_normals ) {
            nn.dir_x = n2.dir_x;
            nn.dir_y = n2.dir_y;
        }

        constexpr std::size_t nb_to_rem = nb_outside - 2;
        for( std::size_t i = i2 + 1; i < si; ++i )
            nodes->local_at( i - nb_to_rem ).get_straight_content_from( nodes->local_at( i ) );

        // modification of the number of points
        size -= nb_to_rem;
        return true;
    }
    case mul_size * 4 + 0b1111:
        size = 0;
        return true;

    // ------------------------------------------------
    case mul_size * 3 + 0b000:
        return false;
    case mul_size * 3 + 0b001:
        TODO;
        return true;
    case mul_size * 3 + 0b010:
        TODO;
        return true;
    case mul_size * 3 + 0b011:
        TODO;
        return true;
    case mul_size * 3 + 0b100:
        TODO;
        return true;
    case mul_size * 3 + 0b101:
        TODO;
        return true;
    case mul_size * 3 + 0b110: {

        return true;
    }
    case mul_size * 3 + 0b111:
        TODO;
        return true;
    default:
        return false;
    }
}

template<class Pc> template<int flags,class BS,class DS>
bool ConvexPolyhedron2<Pc>::plane_cut_gen( Pt origin, Pt normal, CI cut_id, N<flags>, BS &outside, DS &distances ) {
    std::size_t cpt_node = 0;
    for_each_node( [&]( const Node &node ) {
        TF d = dot( origin - node.pos(), normal );
        distances[ cpt_node ] = d;
        outside[ cpt_node ] = d < 0;
        ++cpt_node;
    } );

    // all inside ?
    if ( none( outside ) )
        return false;

    // all outside ?
    unsigned nb_outside = popcnt( outside );
    if ( nb_outside == size ) {
        size = 0;
        return false;
    }

    // => we will need a normalized direction
    if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 ) {
        TF n = 1 / norm_2( normal );
        for( std::size_t i = 0; i < size; ++i )
            distances[ i ] *= n;
        normal = n * normal;
    }

    // => we will need index of the first outside point
    std::size_t i1 = tzcnt( outside );

    // only 1 outside ?
    if ( nb_outside == 1 ) {
        // => creation of a new point
        std::size_t i0 = ( i1 + size - 1 ) % size;
        std::size_t i2 = ( i1 + 1 ) % size;
        std::size_t in = i1 + 1;

        Node &n0 = node( i0 );
        Node &n1 = node( i1 );
        Node &n2 = node( i2 );
        Node &nn = node( in );

        TF s0 = distances[ i0 ];
        TF s1 = distances[ i1 ];
        TF s2 = distances[ i2 ];

        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );

        // shift points
        for( std::size_t i = size; i > i1 + 1; --i )
            node( i + 1 ).get_straight_content_from( node( i + 0 ) );
        ++size;

        // modified or added points
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) {
            nn.dir_x = n1.dir_x;
            nn.dir_y = n1.dir_y;
        }

        n1.x = n0.x - m0 * ( n1.x - n0.x );
        n1.y = n0.y - m0 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) {
            n1.dir_x = normal.x;
            n1.dir_y = normal.y;
        }

        return true;
    }

    // 2 points are outside ?
    if ( nb_outside == 2 ) {
        if ( i1 == 0 && outside[ 1 ] == 0 )
            i1 = size - 1;

        std::size_t i0 = ( i1 + size - 1 ) % size;
        std::size_t i2 = ( i1 + 1 )        % size;
        std::size_t i3 = ( i1 + 2 )        % size;

        Node &n0 = node( i0 );
        Node &n1 = node( i1 );
        Node &n2 = node( i2 );
        Node &n3 = node( i3 );

        TF s0 = distances[ i0 ];
        TF s1 = distances[ i1 ];
        TF s2 = distances[ i2 ];
        TF s3 = distances[ i3 ];

        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );

        // modified points
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        nodes->local_at( 1 ).cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }

    // more than 2 points are outside, outside points are before and after bit 0
    if ( i1 == 0 && outside[ size - 1 ] ) {
        TODO;
        //        std::size_t nb_inside = size - nb_outside;
        //        #ifdef __AVX2__
        //        std::size_t i3 = _tzcnt_u64( ~ outside );
        //        #else
        //        std::size_t i3 = 0;
        //        for( auto cp = ~ outside; ( cp & 1 ) == 0; ++i3 )
        //            cp /= 2;
        //        #endif
        //        i1 = nb_inside + i3;
        //        std::size_t i0 = i1 - 1;
        //        std::size_t i2 = i3 - 1;

        //        Pt p0 { points[ 0 ][ i0 ], points[ 1 ][ i0 ] };
        //        Pt p1 { points[ 0 ][ i1 ], points[ 1 ][ i1 ] };
        //        Pt p2 { points[ 0 ][ i2 ], points[ 1 ][ i2 ] };
        //        Pt p3 { points[ 0 ][ i3 ], points[ 1 ][ i3 ] };
        //        TF s0 = distances[ i0 ];
        //        TF s1 = distances[ i1 ];
        //        TF s2 = distances[ i2 ];
        //        TF s3 = distances[ i3 ];

        //        TF m1 = s0 / ( s1 - s0 );
        //        TF m2 = s3 / ( s2 - s3 );

        //        // modified and shifted points
        //        points[ 0 ][ 0 ] = p3.x - m2 * ( p2.x - p3.x );
        //        points[ 1 ][ 0 ] = p3.y - m2 * ( p2.y - p3.y );
        //        cut_ids    [ 0 ] = cut_ids[ i2 ];
        //        if ( store_the_normals ) {
        //            normals[ 0 ][ 0 ] = normals[ 0 ][ i2 ];
        //            normals[ 1 ][ 0 ] = normals[ 1 ][ i2 ];
        //        }
        //        std::size_t o = 1;
        //        for( ; o <= nb_inside; ++o ) {
        //            points[ 0 ][ o ] = points[ 0 ][ i2 + o ];
        //            points[ 1 ][ o ] = points[ 1 ][ i2 + o ];
        //            cut_ids    [ o ] = cut_ids    [ i2 + o ];
        //            if ( store_the_normals ) {
        //                normals[ 0 ][ o ] = normals[ 0 ][ i2 + o ];
        //                normals[ 1 ][ o ] = normals[ 1 ][ i2 + o ];
        //            }
        //        }
        //        points[ 0 ][ o ] = p0.x - m1 * ( p1.x - p0.x );
        //        points[ 1 ][ o ] = p0.y - m1 * ( p1.y - p0.y );
        //        cut_ids    [ o ] = cut_id;
        //        if ( store_the_normals ) {
        //            normals[ 0 ][ o ] = normal[ 0 ];
        //            normals[ 1 ][ o ] = normal[ 1 ];
        //        }

        //        size -= nb_outside - 2;
        return true;
    }

    // more than 2 points are outside, outside points do not cross `nb_points`
    std::size_t i0 = ( i1 + size - 1       ) % size;
    std::size_t i2 = ( i1 + nb_outside - 1 ) % size;
    std::size_t i3 = ( i1 + nb_outside     ) % size;
    std::size_t in = i1 + 1;

    Node &n0 = node( i0 );
    Node &n1 = node( i1 );
    Node &n2 = node( i2 );
    Node &n3 = node( i3 );
    Node &nn = node( in );

    TF s0 = distances[ i0 ];
    TF s1 = distances[ i1 ];
    TF s2 = distances[ i2 ];
    TF s3 = distances[ i3 ];

    TF m1 = s0 / ( s1 - s0 );
    TF m2 = s3 / ( s2 - s3 );

    // modified and deleted points
    if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
    n1.x = n0.x - m1 * ( n1.x - n0.x );
    n1.y = n0.y - m1 * ( n1.y - n0.y );
    n1.cut_id.set( cut_id );

    if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
    nn.x = n3.x - m2 * ( n2.x - n3.x );
    nn.y = n3.y - m2 * ( n2.y - n3.y );
    nn.cut_id.set( n2.cut_id.get() );

    std::size_t nb_to_rem = nb_outside - 2;
    for( std::size_t i = i2 + 1; i < size; ++i )
        node( i - nb_to_rem ).get_straight_content_from( node( i ) );

    // modification of the number of points
    size -= nb_to_rem;
    return true;
}

template<class Pc> template<int flags>
bool ConvexPolyhedron2<Pc>::plane_cut( Pt origin, Pt normal, CI cut_id, N<flags> ) {
    // no simd ?
    if ( flags & ConvexPolyhedron::do_not_use_simd ) {
        if ( size <= 64 ) {
            std::bitset<64> outside;
            std::array<TF,64> distances;
            return plane_cut_gen( origin, normal, cut_id, N<flags>(), outside, distances );
        }

        std::vector<bool> outside( size );
        std::vector<TF> distances( size );
        return plane_cut_gen( origin, normal, cut_id, N<flags>(), outside, distances );
    }

    // no switch version ?
    if ( flags & ConvexPolyhedron::plane_cut_flag_no_switches )
        return plane_cut_simd_tzcnt( origin, normal, cut_id, N<flags>() );

    // => default version
    return plane_cut_simd_switch( origin, normal, cut_id, N<flags>() );
}

} // namespace sdot
