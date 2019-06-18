#include "../Support/Display/generic_ostream_output.h"
#include "../Support/Display/binary_repr.h"
#include "../Support/aligned_memory.h"
// #include "../Support/xsimd_util.h"
#include "../Support/bit_handling.h"
#include "../Support/ASSERT.h"
#include "../Support/TODO.h"
#include "../Support/P.h"
#include "ConvexPolyhedron2.h"
//#include <xsimd/xsimd.hpp>
#include <cstring>
#include <iomanip>
#include <bitset>

//#define _USE_MATH_DEFINES
//#include <math.h>

#include "Internal/(convex_polyhedron_plane_cut_simd_switch.cpp).h"

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
    nodes = new ( aligned_malloc( rese / block_size * sizeof( Node ), 64 ) ) Node;
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
            std::memcpy( &n.x, &t.x, r * sizeof( TF ) );
            std::memcpy( &n.y, &t.y, r * sizeof( TF ) );
            if ( store_the_normals ) {
                std::memcpy( &n.dir_x, &t.dir_x, r * sizeof( TF ) );
                std::memcpy( &n.dir_y, &t.dir_y, r * sizeof( TF ) );
            }
            if ( allow_ball_cut ) {
                std::memcpy( &n.arc_radius  , &t.arc_radius  , r * sizeof( TF ) );
                std::memcpy( &n.arc_center_x, &t.arc_center_x, r * sizeof( TF ) );
                std::memcpy( &n.arc_center_y, &t.arc_center_y, r * sizeof( TF ) );
            }
            for( std::size_t j = 0; j < r; ++j )
                n.local_at( j ).cut_id.set( t.local_at( j ).cut_id.get() );
            break;
        }

        std::memcpy( &n.x, &t.x, block_size * sizeof( TF ) );
        std::memcpy( &n.y, &t.y, block_size * sizeof( TF ) );
        if ( store_the_normals ) {
            std::memcpy( &n.dir_x, &t.dir_x, block_size * sizeof( TF ) );
            std::memcpy( &n.dir_y, &t.dir_y, block_size * sizeof( TF ) );
        }
        if ( allow_ball_cut ) {
            std::memcpy( &n.arc_radius  , &t.arc_radius  , block_size * sizeof( TF ) );
            std::memcpy( &n.arc_center_x, &t.arc_center_x, block_size * sizeof( TF ) );
            std::memcpy( &n.arc_center_y, &t.arc_center_y, block_size * sizeof( TF ) );
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

        nodes = reinterpret_cast<Node *>( aligned_malloc( nb_blocks * sizeof( Node ), 64 ) );
        for( TI i = 0; i < old_nb_blocks; ++i )
            new ( nodes + i ) Node( std::move( old_nodes[ i ] ) );

        if ( old_nb_blocks )
            aligned_free( old_nodes );
    }
    size = new_size;
}

template<class Pc>
void ConvexPolyhedron2<Pc>::plane_cut( const Cut *cuts, std::size_t nb_cuts ) {
    return plane_cut( cuts, nb_cuts, N<0>() );
}

template<class Pc> template<int flags>
void ConvexPolyhedron2<Pc>::plane_cut_simd_tzcnt( const Cut &cut, N<flags> ) {
    return plane_cut_gen( cut, N<flags>() );

//    constexpr std::size_t simd_size = xsimd::simd_type<TF>::size;
//    using BB = xsimd::batch_bool<TF,simd_size>;
//    using BF = xsimd::batch<TF,simd_size>;

//    // for now this procedure works
//    if ( size > simd_size )
//        return plane_cut_gen( origin, normal, cut_id, N<flags>() );

//    // outsize list
//    BF ox( origin.x );
//    BF oy( origin.y );
//    BF nx( normal.x );
//    BF ny( normal.y );

//    BF px; px.load_aligned( &nodes->x );
//    BF py; py.load_aligned( &nodes->y );
//    BF d = ( ox - px ) * nx + ( oy - py ) * ny;
//    std::uint64_t outside = is_neg( d ) & ( ( 1 << size ) - 1 );

//    // all inside ?
//    if ( outside == 0 )
//        return false;

//    // all outside ?
//    unsigned nb_outside = popcnt( outside );
//    if ( nb_outside == size ) {
//        size = 0;
//        return false;
//    }

//    // => we will need a normalized direction
//    if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 ) {
//        TF n = 1 / norm_2( normal );
//        normal = n * normal;
//        //d *= n;
//    }

//    // => we will need index of the first outside point
//    const TF *distances = reinterpret_cast<const TF *>( &d );
//    std::size_t i1 = tzcnt( outside );

//    // only 1 outside ?
//    if ( nb_outside == 1 ) {
//        // => creation of a new point
//        std::size_t old_size = size;
//        resize( size + 1 );

//        std::size_t i0 = ( i1 + old_size - 1 ) % old_size;
//        std::size_t i2 = ( i1 + 1 ) % old_size;
//        std::size_t in = i1 + 1;

//        Node &n0 = node( i0 );
//        Node &n1 = node( i1 );
//        Node &n2 = node( i2 );
//        Node &nn = node( in );

//        TF s0 = distances[ i0 ];
//        TF s1 = distances[ i1 ];
//        TF s2 = distances[ i2 ];

//        TF m0 = s0 / ( s1 - s0 );
//        TF m1 = s2 / ( s1 - s2 );

//        // save coordinates that can be modified
//        TF n0_x = n0.x;
//        TF n0_y = n0.y;

//        // shift points
//        for( std::size_t i = old_size; i > in; --i )
//            node( i ).get_straight_content_from( node( i - 1 ) );

//        // modified or added points
//        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
//        nn.x = n2.x - m1 * ( n1.x - n2.x );
//        nn.y = n2.y - m1 * ( n1.y - n2.y );
//        nn.cut_id.set( n1.cut_id.get() );

//        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
//        n1.x = n0_x - m0 * ( n1.x - n0_x );
//        n1.y = n0_y - m0 * ( n1.y - n0_y );
//        n1.cut_id.set( cut_id );

//        return true;
//    }

//    // 2 points are outside ?
//    if ( nb_outside == 2 ) {
//        if ( i1 == 0 && ! ( outside & 2 ) )
//            i1 = size - 1;

//        std::size_t i0 = ( i1 + size - 1 ) % size;
//        std::size_t i2 = ( i1 + 1 )        % size;
//        std::size_t i3 = ( i1 + 2 )        % size;

//        Node &n0 = node( i0 );
//        Node &n1 = node( i1 );
//        Node &n2 = node( i2 );
//        Node &n3 = node( i3 );

//        TF s0 = distances[ i0 ];
//        TF s1 = distances[ i1 ];
//        TF s2 = distances[ i2 ];
//        TF s3 = distances[ i3 ];

//        TF m1 = s0 / ( s1 - s0 );
//        TF m2 = s3 / ( s2 - s3 );

//        // modified points
//        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
//        n1.cut_id.set( cut_id );
//        n1.x = n0.x - m1 * ( n1.x - n0.x );
//        n1.y = n0.y - m1 * ( n1.y - n0.y );
//        n2.x = n3.x - m2 * ( n2.x - n3.x );
//        n2.y = n3.y - m2 * ( n2.y - n3.y );
//        return true;
//    }

//    // more than 2 points are outside, outside points are before and after bit 0
//    if ( i1 == 0 && ( outside & ( 1ul << ( size - 1 ) ) ) ) {
//        std::size_t nb_inside = size - nb_outside;
//        std::size_t i3 = tocnt( outside );
//        i1 = nb_inside + i3;

//        std::size_t i0 = i1 - 1;
//        std::size_t i2 = i3 - 1;
//        std::size_t in = 0;

//        Node &n0 = node( i0 );
//        Node &n1 = node( i1 );
//        Node &n2 = node( i2 );
//        Node &n3 = node( i3 );
//        Node &nn = node( in );

//        TF s0 = distances[ i0 ];
//        TF s1 = distances[ i1 ];
//        TF s2 = distances[ i2 ];
//        TF s3 = distances[ i3 ];

//        TF m1 = s0 / ( s1 - s0 );
//        TF m2 = s3 / ( s2 - s3 );

//        // modified and shifted points
//        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
//        nn.x = n3.x - m2 * ( n2.x - n3.x );
//        nn.y = n3.y - m2 * ( n2.y - n3.y );
//        nn.cut_id.set( n2.cut_id.get() );

//        std::size_t o = 1;
//        for( ; o <= nb_inside; ++o )
//            nodes->local_at( o ).get_straight_content_from( nodes->local_at( i2 + o ) );
//        Node &no = node( o );

//        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
//        no.x = n0.x - m1 * ( n1.x - n0.x );
//        no.y = n0.y - m1 * ( n1.y - n0.y );
//        no.cut_id.set( cut_id );

//        size -= nb_outside - 2;
//        return true;
//    }

//    // more than 2 points are outside, outside points do not cross `nb_points`
//    std::size_t i0 = ( i1 + size - 1       ) % size;
//    std::size_t i2 = ( i1 + nb_outside - 1 ) % size;
//    std::size_t i3 = ( i1 + nb_outside     ) % size;
//    std::size_t in = i1 + 1;

//    Node &n0 = node( i0 );
//    Node &n1 = node( i1 );
//    Node &n2 = node( i2 );
//    Node &n3 = node( i3 );
//    Node &nn = node( in );

//    TF s0 = distances[ i0 ];
//    TF s1 = distances[ i1 ];
//    TF s2 = distances[ i2 ];
//    TF s3 = distances[ i3 ];

//    TF m1 = s0 / ( s1 - s0 );
//    TF m2 = s3 / ( s2 - s3 );

//    // modified and deleted points
//    if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
//    n1.x = n0.x - m1 * ( n1.x - n0.x );
//    n1.y = n0.y - m1 * ( n1.y - n0.y );
//    n1.cut_id.set( cut_id );

//    if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
//    nn.x = n3.x - m2 * ( n2.x - n3.x );
//    nn.y = n3.y - m2 * ( n2.y - n3.y );
//    nn.cut_id.set( n2.cut_id.get() );

//    std::size_t nb_to_rem = nb_outside - 2;
//    for( std::size_t i = i2 + 1; i < size; ++i )
//        nodes->local_at( i - nb_to_rem ).get_straight_content_from( nodes->local_at( i ) );

//    // modification of the number of points
//    size -= nb_to_rem;
//    return true;
}

template<class Pc> template<int flags,class BS,class DS>
void ConvexPolyhedron2<Pc>::plane_cut_gen( const Cut &cut, N<flags>, BS &outside, DS &distances ) {
    reset( outside );
    std::size_t cpt_node = 0;
    for_each_node( [&]( const Node &node ) {
        TF d = dot( node.pos(), cut.dir );
        outside[ cpt_node ] = d > cut.dist;
        distances[ cpt_node ] = d - cut.dist;
        ++cpt_node;
    } );

    // all inside ?
    if ( none( outside ) )
        return;

    // all outside ?
    unsigned nb_outside = popcnt( outside );
    if ( nb_outside == size ) {
        size = 0;
        return;
    }

    // => we will need a normalized direction
    //        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
    //            dir = normalized( dir );

    // => we will need index of the first outside point
    std::size_t i1 = tzcnt( outside );

    // only 1 outside ?
    if ( nb_outside == 1 ) {
        // => creation of a new point
        std::size_t old_size = size;
        resize( size + 1 );

        std::size_t i0 = ( i1 + old_size - 1 ) % old_size;
        std::size_t i2 = ( i1 + 1 ) % old_size;
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

        // save coordinates that can be modified
        TF n0_x = n0.x;
        TF n0_y = n0.y;

        // shifted points
        for( std::size_t i = old_size; i > in; --i )
            node( i ).get_straight_content_from( node( i - 1 ) );

        // modified or added points
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );

        //        if ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized == 0 )
        //            dir = normalized( dir );
        if ( store_the_normals ) {  n1.dir_x = cut.dir.x; n1.dir_y = cut.dir.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut.id );

        return;
    }

    // 2 points are outside ?
    if ( nb_outside == 2 ) {
        if ( i1 == 0 && ! outside[ 1 ] )
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
        if ( store_the_normals ) { n1.dir_x = cut.dir.x; n1.dir_y = cut.dir.y; }
        n1.cut_id.set( cut.id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );

        return;
    }

    // more than 2 points are outside, outside points are before and after bit 0
    if ( i1 == 0 && outside[ size - 1 ] ) {
        std::size_t nb_inside = size - nb_outside;
        std::size_t i3 = tocnt( outside );
        i1 = nb_inside + i3;

        std::size_t i0 = i1 - 1;
        std::size_t i2 = i3 - 1;
        std::size_t in = 0;

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

        // modified and shifted points
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );

        std::size_t o = 1;
        for( ; o <= nb_inside; ++o )
            node( o ).get_straight_content_from( node( i2 + o ) );
        Node &no = node( o );

        if ( store_the_normals ) { no.dir_x = cut.dir.x; no.dir_y = cut.dir.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut.id );

        size -= nb_outside - 2;

        return;
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
    if ( store_the_normals ) { n1.dir_x = cut.dir.x; n1.dir_y = cut.dir.y; }
    n1.x = n0.x - m1 * ( n1.x - n0.x );
    n1.y = n0.y - m1 * ( n1.y - n0.y );
    n1.cut_id.set( cut.id );

    if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
    nn.x = n3.x - m2 * ( n2.x - n3.x );
    nn.y = n3.y - m2 * ( n2.y - n3.y );
    nn.cut_id.set( n2.cut_id.get() );

    std::size_t nb_to_rem = nb_outside - 2;
    for( std::size_t i = i2 + 1; i < size; ++i )
        node( i - nb_to_rem ).get_straight_content_from( node( i ) );

    // modification of the number of points
    size -= nb_to_rem;
}

template<class Pc> template<int flags>
void ConvexPolyhedron2<Pc>::plane_cut_gen( const Cut &cut, N<flags> ) {
    if ( size <= 64 ) {
        std::bitset<64> outside;
        std::array<TF,64> distances;
        return plane_cut_gen( cut, N<flags>(), outside, distances );
    }

    std::vector<TF> distances( size );
    std::vector<bool> outside( size );
    return plane_cut_gen( cut, N<flags>(), outside, distances );
}

template<class Pc> template<int flags>
void ConvexPolyhedron2<Pc>::plane_cut( const Cut *cuts, std::size_t nb_cuts, N<flags> ) {
    // no simd ?
    if ( flags & ConvexPolyhedron::do_not_use_simd )
        for( std::size_t i = 0; i < nb_cuts; ++i )
            return plane_cut_gen( cuts[ i ], N<flags>() );

    // no switch version ?
    if ( flags & ConvexPolyhedron::do_not_use_switches )
        for( std::size_t i = 0; i < nb_cuts; ++i )
            return plane_cut_simd_tzcnt( cuts[ i ], N<flags>() );

    // => default version
    return plane_cut_simd_switch( cuts, nb_cuts, N<flags>(), S<TF>() );
}

} // namespace sdot
