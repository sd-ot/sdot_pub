#include "../Support/Display/generic_ostream_output.h"
#include "../Support/Display/binary_repr.h"
#include "../Support/aligned_memory.h"
#include "../Support/bit_handling.h"
#include "../Support/ASSERT.h"
#include "../Support/TODO.h"
#include "../Support/pi.h"
#include "../Support/P.h"
#include "ConvexPolyhedron3.h"
#include <cstring>
#include <iomanip>
#include <bitset>

//#define _USE_MATH_DEFINES
//#include <math.h>

// #include "Internal/(convex_polyhedron_plane_cut_simd_switch.cpp).h"

namespace sdot {

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3( const Box &box, CI cut_id ) : ConvexPolyhedron3() {
    set_nb_nodes(  8 );
    set_nb_edges( 12 );

    Node &n0 = node( 0 );
    Node &n1 = node( 1 );
    Node &n2 = node( 2 );
    Node &n3 = node( 3 );
    Node &n4 = node( 4 );
    Node &n5 = node( 5 );
    Node &n6 = node( 6 );
    Node &n7 = node( 7 );

    auto set_node = [&]( TI index, TF x, TF y, TF z ) { node( index ).x = x; node( index ).y = y; node( index ).z = z; };

    set_node( 0, box.p0.x, box.p0.y, box.p0.z );
    set_node( 1, box.p1.x, box.p0.y, box.p0.z );
    set_node( 2, box.p0.x, box.p1.y, box.p0.z );
    set_node( 3, box.p1.x, box.p1.y, box.p0.z );
    set_node( 4, box.p0.x, box.p0.y, box.p1.z );
    set_node( 5, box.p1.x, box.p0.y, box.p1.z );
    set_node( 6, box.p0.x, box.p1.y, box.p1.z );
    set_node( 7, box.p1.x, box.p1.y, box.p1.z );

    auto set_edge = [&]( TI index, TI n0, TI n1 ) { edge( index ).node_0 = n0; edge( index ).node_1 = n1; };

    set_edge(  0, 0, 1 );
    set_edge(  1, 1, 3 );
    set_edge(  2, 3, 2 );
    set_edge(  3, 2, 0 );

    set_edge(  4, 4, 6 );
    set_edge(  5, 6, 7 );
    set_edge(  6, 7, 5 );
    set_edge(  7, 5, 4 );

    set_edge(  8, 0, 4 );
    set_edge(  9, 1, 5 );
    set_edge( 10, 3, 7 );
    set_edge( 11, 2, 6 );

    auto add_face = [&]( int e0, int e1, int e2, int e3 ) {
        Pt P0 = node( edge_n0( e0 ) ).pos();
        Pt P1 = node( edge_n1( e0 ) ).pos();
        Pt P2 = node( edge_n1( e1 ) ).pos();

        Face face;
        if ( allow_ball_cut )
            face.round = false;

        face.num_in_edge_beg = num_in_edges_m2.size();
        face.num_in_edge_len = 4;
        face.normal          = normalized( cross_prod( P0 - P1, P2 - P1 ) );
        face.cut_id          = cut_id;

        faces.push_back( face );

        num_in_edges_m2.push_back( e0 );
        num_in_edges_m2.push_back( e1 );
        num_in_edges_m2.push_back( e2 );
        num_in_edges_m2.push_back( e3 );

        edge( e0 / 2 ).set_face( e0 % 2, faces.size() );
        edge( e1 / 2 ).set_face( e1 % 2, faces.size() );
        edge( e2 / 2 ).set_face( e2 % 2, faces.size() );
        edge( e3 / 2 ).set_face( e3 % 2, faces.size() );
    };

    const int a = 10;
    const int b = 11;

    add_face( 2 * 0 + 0, 2 * 1 + 0, 2 * 2 + 0, 2 * 3 + 0 );
    add_face( 2 * 4 + 0, 2 * 5 + 0, 2 * 6 + 0, 2 * 7 + 0 );

    add_face( 2 * 8 + 0, 2 * 7 + 1, 2 * 9 + 1, 2 * 0 + 1 );
    add_face( 2 * a + 0, 2 * 5 + 1, 2 * b + 1, 2 * 2 + 1 );

    add_face( 2 * b + 0, 2 * 4 + 1, 2 * 8 + 1, 2 * 3 + 1 );
    add_face( 2 * 9 + 0, 2 * 6 + 1, 2 * a + 1, 2 * 1 + 1 );

    //    if ( keep_min_max_coords )
    //        update_min_max_coord();
}

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3() {
    nodes_size  = 0;
    nodes_rese  = block_size;
    nodes = new ( aligned_malloc( nodes_rese / block_size * sizeof( Node ), 64 ) ) Node;

    edges_size  = 0;
    edges_rese  = block_size;
    edges = new ( aligned_malloc( edges_rese / block_size * sizeof( Edge ), 64 ) ) Edge;

    sphere_radius = 0;
}

template<class Pc>
ConvexPolyhedron3<Pc>::~ConvexPolyhedron3() {
    if ( nodes_rese )
        aligned_free( nodes );
}

template<class Pc>
ConvexPolyhedron3<Pc> &ConvexPolyhedron3<Pc>::operator=( const ConvexPolyhedron3 &that ) {
    //    this->nb_changes = that.nb_changes;
    set_nb_nodes( that.nb_nodes() );

    for( std::size_t i = 0; ; i += block_size ) {
        Node &n = nodes[ i / block_size ];
        const Node &t = that.nodes[ i / block_size ];
        if ( i + block_size > nodes_size ) {
            std::size_t r = nodes_size - i;
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
    sphere_cut_id     = that.sphere_cut_id;

    return *this;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::write_to_stream( std::ostream &os ) const {
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
void ConvexPolyhedron3<Pc>::display( VtkOutput &vo, const std::vector<TF> &cell_values, Pt offset, bool display_both_sides ) const {
    std::vector<VtkOutput::Pt> pts;
    for( const Face &face : faces ) {
        if ( face.round ) {
            TODO;
        } else if ( display_both_sides || face.cut_id > sphere_cut_id ) {
            pts.resize( face.num_in_edge_len );
            for( std::size_t i = 0; i < face.num_in_edge_len; ++i ) {
                int num_in_edge = num_in_edges_m2[ face.num_in_edge_beg + i ];
                pts[ i ] = node( edge_n0( num_in_edge ) ).pos();
            }
            vo.add_polygon( pts, cell_values );
        }
    }
}

template<class Pc> template<class F>
void ConvexPolyhedron3<Pc>::for_each_edge( const F &/*f*/ ) const {
    TODO;
}

template<class Pc> template<class F>
void ConvexPolyhedron3<Pc>::for_each_node( const F &f ) const {
    static_assert ( sizeof( Node ) % sizeof( TF ) == 0, "" );

    Node *ptr = nodes;
    for( TI i = 0; ; ++i ) {
        if ( i + block_size >= nodes_size ) {
            for( TI j = 0; j < nodes_size - i; ++j )
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
typename ConvexPolyhedron3<Pc>::TI ConvexPolyhedron3<Pc>::nb_nodes() const {
    return nodes_size;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TI ConvexPolyhedron3<Pc>::nb_edges() const {
    return edges_size;
}

template<class Pc>
bool ConvexPolyhedron3<Pc>::empty() const {
    return nodes_size == 0;
}

template<class Pc>
const typename ConvexPolyhedron3<Pc>::Node &ConvexPolyhedron3<Pc>::node( TI index ) const {
    return nodes->global_at( index );
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::Node &ConvexPolyhedron3<Pc>::node( TI index ) {
    return nodes->global_at( index );
}

template<class Pc>
const typename ConvexPolyhedron3<Pc>::Edge &ConvexPolyhedron3<Pc>::edge( TI index ) const {
    return edges->global_at( index );
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::Edge &ConvexPolyhedron3<Pc>::edge( TI index ) {
    return edges->global_at( index );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::for_each_boundary_item( const std::function<void( const BoundaryItem &boundary_item )> &/*f*/, TF /*weight*/ ) const {
    TODO;
    //    if ( nb_nodes() == 0 ) {
    //        if ( sphere_radius >= 0 ) {
    //            BoundaryItem item;
    //            item.id = sphere_cut_id;
    //            item.measure = 2 * pi( S<TF>() ) * sphere_radius;
    //            item.a0 = 1;
    //            item.a1 = 0;
    //            f( item );
    //        }
    //        return;
    //    }

    //    for( size_t i1 = 0, i0 = nb_nodes() - 1; i1 < nb_nodes(); i0 = i1++ ) {
    //        BoundaryItem item;
    //        //item.id = node( i0 ).cut_id.get();
    //        item.points[ 0 ] = node( i0 ).pos();
    //        item.points[ 1 ] = node( i1 ).pos();

    //        if ( allow_ball_cut && node( i0 ).arc_radius > 0 ) {
    //            using std::atan2;
    //            item.a0 = atan2( node( i0 ).y - sphere_center.y, node( i0 ).x - sphere_center.x );
    //            item.a1 = atan2( node( i1 ).y - sphere_center.y, node( i1 ).x - sphere_center.x );
    //            if ( item.a1 < item.a0 )
    //                item.a1 += 2 * pi( S<TF>() );
    //            item.measure = ( item.a1 - item.a0 ) * sphere_radius;
    //        } else {
    //            item.measure = norm_2( node( i1 ).pos() - node( i0 ).pos() );
    //        }

    //        f( item );
    //    }
}

template<class Pc>
void ConvexPolyhedron3<Pc>::set_nb_nodes( TI new_size ) {
    if ( new_size > nodes_rese ) {
        TI old_nb_blocks = nodes_rese / block_size;
        Node *old_nodes = nodes;

        TI nb_blocks = ( new_size + block_size - 1 ) / block_size;
        nodes_rese = nb_blocks * block_size;

        nodes = reinterpret_cast<Node *>( aligned_malloc( nb_blocks * sizeof( Node ), 64 ) );
        for( TI i = 0; i < old_nb_blocks; ++i )
            new ( nodes + i ) Node( std::move( old_nodes[ i ] ) );

        if ( old_nb_blocks )
            aligned_free( old_nodes );
    }
    nodes_size = new_size;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::set_nb_edges( TI new_size ) {
    if ( new_size > edges_rese ) {
        TI old_nb_blocks = edges_rese / block_size;
        Edge *old_edges = edges;

        TI nb_blocks = ( new_size + block_size - 1 ) / block_size;
        edges_rese = nb_blocks * block_size;

        edges = reinterpret_cast<Edge *>( aligned_malloc( nb_blocks * sizeof( Edge ), 64 ) );
        for( TI i = 0; i < old_nb_blocks; ++i )
            new ( edges + i ) Edge( std::move( old_edges[ i ] ) );

        if ( old_nb_blocks )
            aligned_free( old_edges );
    }
    edges_size = new_size;
}

template<class Pc> template<int flags>
void ConvexPolyhedron3<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ) {
}

template<class Pc>
void ConvexPolyhedron3<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ) {
    return plane_cut( cut_dir, cut_ps, cut_id, nb_cuts, N<0>() );
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::integral() const {
    TODO;
    return 0;
}

template<class Pc> template<class TL>
void ConvexPolyhedron3<Pc>::BoundaryItem::add_simplex_list( TL &/*lst*/ ) const {
    // lst.push_back( { points[ 0 ], points[ 1 ] } );
}

} // namespace sdot
