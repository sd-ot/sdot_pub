#include "../Support/Display/generic_ostream_output.h"
#include "../Support/Display/binary_repr.h"
#include "../Support/aligned_memory.h"
// #include "../Support/xsimd_util.h"
#include "../Support/bit_handling.h"
#include "../Support/ASSERT.h"
#include "../Support/TODO.h"
#include "../Support/pi.h"
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
ConvexPolyhedron2<Pc>::ConvexPolyhedron2( ConvexPolyhedron2 &&that ) {
    nodes         = that.nodes        ;
    size          = that.size         ;
    rese          = that.rese         ;

    sphere_radius = that.sphere_radius;
    sphere_center = that.sphere_center;
    sphere_cut_id = that.sphere_cut_id;

    that.nodes    = nullptr;
    that.size     = 0;
    that.rese     = 0;
}

template<class Pc>
ConvexPolyhedron2<Pc>::ConvexPolyhedron2() {
    size  = 0;
    rese  = block_size;
    nodes = new ( aligned_malloc( rese / block_size * sizeof( Node ), 64 ) ) Node;

    sphere_radius = 0;
}

template<class Pc>
ConvexPolyhedron2<Pc>::~ConvexPolyhedron2() {
    if ( rese )
        aligned_free( nodes );
}

template<class Pc>
ConvexPolyhedron2<Pc> &ConvexPolyhedron2<Pc>::operator=( const ConvexPolyhedron2 &that ) {
    //    this->nb_changes = that.nb_changes;
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

    sphere_radius = that.sphere_radius;
    sphere_center = that.sphere_center;
    sphere_cut_id = that.sphere_cut_id;

    return *this;
}

template<class Pc>
void ConvexPolyhedron2<Pc>::write_to_stream( std::ostream &os ) const {
    os << "pos: ";
    for( TI i = 0; i < nb_nodes(); ++i )
        os << ( i ? " [" : "[" ) << node( i ).pos().x << " " << node( i ).pos().y << "]";
    //    if ( store_the_normals ) {
    //        os << " nrms: ";
    //        for( TI i = 0; i < nb_nodes(); ++i )
    //            os << ( i ? " [" : "[" ) << node( i ).dir() << "]";
    //    }
    //    os << " sphere center: " << sphere_center << " sphere radius: " << sphere_radius;
}

template<class Pc>
void ConvexPolyhedron2<Pc>::display_vtk( VtkOutput &vo, const std::vector<TF> &cell_values, Pt offset ) const {
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
bool ConvexPolyhedron2<Pc>::empty() const {
    return size == 0;
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
void ConvexPolyhedron2<Pc>::for_each_boundary_item( const std::function<void( const BoundaryItem &boundary_item )> &f, TF /*weight*/ ) const {
    if ( nb_nodes() == 0 ) {
        if ( sphere_radius >= 0 ) {
            BoundaryItem item;
            item.dirac = sphere_cut_id;
            item.measure = 2 * pi( S<TF>() ) * sphere_radius;
            item.a0 = 1;
            item.a1 = 0;
            f( item );
        }
        return;
    }

    for( size_t i1 = 0, i0 = nb_nodes() - 1; i1 < nb_nodes(); i0 = i1++ ) {
        BoundaryItem item;
        item.dirac = node( i0 ).cut_id.get();
        item.points[ 0 ] = node( i0 ).pos();
        item.points[ 1 ] = node( i1 ).pos();

        if ( allow_ball_cut && node( i0 ).arc_radius > 0 ) {
            using std::atan2;
            item.a0 = atan2( node( i0 ).y - sphere_center[ 1 ], node( i0 ).x - sphere_center[ 0 ] );
            item.a1 = atan2( node( i1 ).y - sphere_center[ 1 ], node( i1 ).x - sphere_center[ 0 ] );
            if ( item.a1 < item.a0 )
                item.a1 += 2 * pi( S<TF>() );
            item.measure = ( item.a1 - item.a0 ) * sphere_radius;
        } else {
            item.measure = norm_2( node( i1 ).pos() - node( i0 ).pos() );
        }

        f( item );
    }
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

template<class Pc> template<int flags,class T,class U>
void ConvexPolyhedron2<Pc>::plane_cut_simd_tzcnt( TF cut_dx, TF cut_dy, TF cut_ps, CI cut_id, N<flags>, S<T>, S<U> ) {
    plane_cut_gen( cut_dx, cut_dy, cut_ps, cut_id, N<flags>() );
}

template<class Pc> template<int flags>
void ConvexPolyhedron2<Pc>::plane_cut_simd_tzcnt( TF cut_dx, TF cut_dy, TF cut_ps, CI cut_id, N<flags>, S<double>, S<std::uint64_t> ) {
    #ifdef __AVX512F__
    constexpr int simd_size = 8;

    // for now this procedure works for 1 simd register
    if ( size > simd_size )
        return plane_cut_gen( cut_dx, cut_dy, cut_ps, cut_id, N<flags>() );

    __m512d rd = _mm512_set1_pd( cut_ps );
    __m512d nx = _mm512_set1_pd( cut_dx );
    __m512d ny = _mm512_set1_pd( cut_dy );
    __m512d px_0 = _mm512_load_pd( &nodes->x + 0 );
    __m512d py_0 = _mm512_load_pd( &nodes->y + 0 );
    __m512d bi_0 = _mm512_add_pd( _mm512_mul_pd( px_0, nx ), _mm512_mul_pd( py_0, ny ) );
    std::uint8_t outside_0 = _mm512_cmp_pd_mask( bi_0, rd, _CMP_GT_OQ ) & ( ( 1 << size ) - 1 );
    __m512d di_0 = _mm512_sub_pd( bi_0, rd );

    // all inside ?
    if ( outside_0 == 0 )
        return;

    // reg change
    //    if ( flags & ConvexPolyhedron::get_nb_changes )
    //        ++this->nb_changes;

    // all outside ?
    unsigned nb_outside = popcnt( outside_0 );
    if ( nb_outside == size ) {
        size = 0;
        return;
    }

    // => we will need index of the first outside point
    const TF *distances = reinterpret_cast<const TF *>( &di_0 );
    std::size_t i1 = tzcnt( outside_0 );

    // only 1 outside ?
    if ( nb_outside == 1 ) {
        // => creation of a new point
        std::size_t old_size = size;
        resize( size + 1 );

        std::size_t i0 = ( i1 + old_size - 1 ) % old_size;
        std::size_t i2 = ( i1 + 1 ) % old_size;
        std::size_t in = i1 + 1;

        Node &n0 = nodes->local_at( i0 );
        Node &n1 = nodes->local_at( i1 );
        Node &n2 = nodes->local_at( i2 );
        Node &nn = nodes->local_at( in );

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
            nodes->local_at( i ).get_straight_content_from( nodes->local_at( i - 1 ) );

        // modified or added points
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );

        //        if ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized == 0 )
        //            dir = normalized( dir );
        if ( store_the_normals ) {  n1.dir_x = cut_dx; n1.dir_y = cut_dy; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );

        return;
    }

    // 2 points are outside ?
    if ( nb_outside == 2 ) {
        if ( i1 == 0 && ! ( outside_0 & 2 ) )
            i1 = size - 1;

        std::size_t i0 = ( i1 + size - 1 ) % size;
        std::size_t i2 = ( i1 + 1 )        % size;
        std::size_t i3 = ( i1 + 2 )        % size;

        Node &n0 = nodes->local_at( i0 );
        Node &n1 = nodes->local_at( i1 );
        Node &n2 = nodes->local_at( i2 );
        Node &n3 = nodes->local_at( i3 );

        TF s0 = distances[ i0 ];
        TF s1 = distances[ i1 ];
        TF s2 = distances[ i2 ];
        TF s3 = distances[ i3 ];

        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );

        // modified points
        if ( store_the_normals ) { n1.dir_x = cut_dx; n1.dir_y = cut_dy; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );

        return;
    }

    // more than 2 points are outside, outside points are before and after bit 0
    if ( i1 == 0 && ( outside_0 & ( 1 << ( size - 1 ) ) ) ) {
        std::size_t nb_inside = size - nb_outside;
        std::size_t i3 = tocnt( outside_0 );
        i1 = nb_inside + i3;

        std::size_t i0 = i1 - 1;
        std::size_t i2 = i3 - 1;
        std::size_t in = 0;

        Node &n0 = nodes->local_at( i0 );
        Node &n1 = nodes->local_at( i1 );
        Node &n2 = nodes->local_at( i2 );
        Node &n3 = nodes->local_at( i3 );
        Node &nn = nodes->local_at( in );

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
            nodes->local_at( o ).get_straight_content_from( nodes->local_at( i2 + o ) );
        Node &no = nodes->local_at( o );

        if ( store_the_normals ) { no.dir_x = cut_dx; no.dir_y = cut_dy; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );

        size -= nb_outside - 2;

        return;
    }

    // more than 2 points are outside, outside points do not cross `nb_points`
    std::size_t i0 = ( i1 + size - 1       ) % size;
    std::size_t i2 = ( i1 + nb_outside - 1 ) % size;
    std::size_t i3 = ( i1 + nb_outside     ) % size;
    std::size_t in = i1 + 1;

    Node &n0 = nodes->local_at( i0 );
    Node &n1 = nodes->local_at( i1 );
    Node &n2 = nodes->local_at( i2 );
    Node &n3 = nodes->local_at( i3 );
    Node &nn = nodes->local_at( in );

    TF s0 = distances[ i0 ];
    TF s1 = distances[ i1 ];
    TF s2 = distances[ i2 ];
    TF s3 = distances[ i3 ];

    TF m1 = s0 / ( s1 - s0 );
    TF m2 = s3 / ( s2 - s3 );

    // modified and deleted points
    if ( store_the_normals ) { n1.dir_x = cut_dx; n1.dir_y = cut_dy; }
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
    #else
    plane_cut_gen( cut_dx, cut_dy, cut_ps, cut_id, N<flags>() );
    #endif
}

template<class Pc> template<int flags,class BS,class DS>
void ConvexPolyhedron2<Pc>::plane_cut_gen( TF cut_dx, TF cut_dy, TF cut_ps, CI cut_id, N<flags>, BS &outside, DS &distances ) {
    reset( outside );
    std::size_t cpt_node = 0;
    for_each_node( [&]( const Node &node ) {
        TF d = node.x * cut_dx + node.y * cut_dy;
        outside[ cpt_node ] = d > cut_ps;
        distances[ cpt_node ] = d - cut_ps;
        ++cpt_node;
    } );

    // all inside ?
    if ( none( outside ) )
        return;

    // reg change
    //    if ( flags & ConvexPolyhedron::get_nb_changes )
    //        ++this->nb_changes;

    // all outside ?
    unsigned nb_outside = popcnt( outside );
    if ( nb_outside == size ) {
        size = 0;
        return;
    }

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
        if ( store_the_normals ) {  n1.dir_x = cut_dx; n1.dir_y = cut_dy; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );

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
        if ( store_the_normals ) { n1.dir_x = cut_dx; n1.dir_y = cut_dy; }
        n1.cut_id.set( cut_id );
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

        if ( store_the_normals ) { no.dir_x = cut_dx; no.dir_y = cut_dy; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );

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
    if ( store_the_normals ) { n1.dir_x = cut_dx; n1.dir_y = cut_dy; }
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
}

template<class Pc> template<int flags>
void ConvexPolyhedron2<Pc>::plane_cut_gen( TF cut_dx, TF cut_dy, TF cut_ps, CI cut_id, N<flags> ) {
    if ( size <= 64 ) {
        std::bitset<64> outside;
        std::array<TF,64> distances;
        return plane_cut_gen( cut_dx, cut_dy, cut_ps, cut_id, N<flags>(), outside, distances );
    }

    std::vector<TF> distances( size );
    std::vector<bool> outside( size );
    return plane_cut_gen( cut_dx, cut_dy, cut_ps, cut_id, N<flags>(), outside, distances );
}

template<class Pc> template<int flags>
void ConvexPolyhedron2<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ) {
    // no simd ?
    if ( flags & ConvexPolyhedron::do_not_use_simd ) {
        for( std::size_t i = 0; i < nb_cuts; ++i )
            plane_cut_gen( cut_dir[ 0 ][ i ], cut_dir[ 1 ][ i ], cut_ps[ i ], cut_id[ i ], N<flags>() );
        return;
    }

    // no switch version ?
    if ( flags & ConvexPolyhedron::do_not_use_switch ) {
        for( std::size_t i = 0; i < nb_cuts; ++i )
            plane_cut_simd_tzcnt( cut_dir[ 0 ][ i ], cut_dir[ 1 ][ i ], cut_ps[ i ], cut_id[ i ], N<flags>(), S<TF>(), S<CI>() );
        return;
    }

    // => default version
    plane_cut_simd_switch( cut_dir, cut_ps, cut_id, nb_cuts, N<flags>(), S<TF>(), S<CI>() );
}

template<class Pc>
void ConvexPolyhedron2<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ) {
    return plane_cut( cut_dir, cut_ps, cut_id, nb_cuts, N<0>() );
}

template<class Pc>
typename ConvexPolyhedron2<Pc>::TF ConvexPolyhedron2<Pc>::integral() const {
    //    // nsmake run -g3 src/PowerDiagram/offline_integration/gen_approx_integration.cpp --function Unit --end-log-scale 100 --precision 1e-10 -r 100 -l 100
    //    if ( _cuts.empty() )
    //        return _sphere_radius > 0 ? 2 * pi() * 0.5 * pow( _sphere_radius, 2 ) : 0;

    //    auto arc_val = []( PT P0, PT P1 ) {
    //        using std::atan2;
    //        using std::pow;
    //        TF a0 = atan2( P0.y, P0.x );
    //        TF a1 = atan2( P1.y, P1.x );
    //        if ( a1 < a0 )
    //            a1 += 2 * pi();
    //        return ( a1 - a0 ) * 0.5 * pow( dot( P0, P0 ), 1 );
    //    };

    //    auto seg_val = []( PT P0, PT P1 ) {
    //        return -0.25 * ( ( P0.x + P1.x ) * ( P0.y - P1.y ) - ( P0.x - P1.x ) * ( P0.y + P1.y ) );
    //    };

    //    TF res = 0;
    //    for( size_t i1 = 0, i0 = _cuts.size() - 1; i1 < _cuts.size(); i0 = i1++ ) {
    //        if ( _cuts[ i0 ].seg_type == SegType::arc )
    //            res += arc_val( _cuts[ i0 ].point - _sphere_center, _cuts[ i1 ].point - _sphere_center );
    //        else
    //            res += seg_val( _cuts[ i0 ].point - _sphere_center, _cuts[ i1 ].point - _sphere_center );
    //    }
    //    return res;

    // hand coded version:
    if ( nb_nodes() == 0 )
        return allow_ball_cut ? pi( S<TF>() ) * pow( sphere_radius, 2 ) : TF( 0 );

    // triangles
    TF res = 0;
    Pt A = node( 0 ).pos();
    for( std::size_t i = 2; i < nb_nodes(); ++i ) {
        Pt B = node( i - 1 ).pos(), C = node( i - 0 ).pos();
        TF tr2_area = A.x * ( B.y - C.y ) + B.x * ( C.y - A.y ) + C.x * ( A.y - B.y );
        res += tr2_area;
    }
    res *= 0.5;

    // arcs
    if ( allow_ball_cut ) {
        TODO;
        //        for( size_t i0 = nb_nodes() - 1, i1 = 0; i1 < nb_nodes(); i0 = i1++ )
        //            if ( arcs[ i0 ] )
        //                res += _arc_area( point( i0 ), point( i1 ) );
    }

    return res;
}

template<class Pc> template<class TL>
void ConvexPolyhedron2<Pc>::BoundaryItem::add_to_simplex_list( TL &lst ) const {
    lst.push_back( { points[ 0 ], points[ 1 ] } );
}

} // namespace sdot
