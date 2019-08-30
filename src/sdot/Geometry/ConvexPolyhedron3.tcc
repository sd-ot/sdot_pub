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
    // nodes
    auto set_node = [&]( TI index, TF x, TF y, TF z ) -> Node * {
        Node *res = &node( index );
        for( TI i = 0; i < 3; ++i )
            res->next_in_faces[ i ].set( { nullptr, 0 } );
        res->x = x;
        res->y = y;
        res->z = z;

        return res;
    };

    set_nb_nodes( 8 );

    Node *n0 = set_node( 0, box.p0.x, box.p0.y, box.p0.z );
    Node *n1 = set_node( 1, box.p1.x, box.p0.y, box.p0.z );
    Node *n2 = set_node( 2, box.p0.x, box.p1.y, box.p0.z );
    Node *n3 = set_node( 3, box.p1.x, box.p1.y, box.p0.z );
    Node *n4 = set_node( 4, box.p0.x, box.p0.y, box.p1.z );
    Node *n5 = set_node( 5, box.p1.x, box.p0.y, box.p1.z );
    Node *n6 = set_node( 6, box.p0.x, box.p1.y, box.p1.z );
    Node *n7 = set_node( 7, box.p1.x, box.p1.y, box.p1.z );

    // faces
    auto add_face = [&]( Node *n0, Node *n1, Node *n2, Node *n3 ) {
        Pt P0 = n0->pos();
        Pt P1 = n1->pos();
        Pt P2 = n2->pos();

        Face *face = faces.create();
        if ( allow_ball_cut )
            face->round = false;
        face->normal = normalized( cross_prod( P0 - P1, P2 - P1 ) );
        face->cut_id = cut_id;

        TI o0 = n0->free_edge();
        TI o1 = n1->free_edge();
        TI o2 = n2->free_edge();
        TI o3 = n3->free_edge();

        face->first_edge = { n0, o0 };
        n0->next_in_faces[ o0 ].set( { n1, o1 } );
        n1->next_in_faces[ o1 ].set( { n2, o2 } );
        n2->next_in_faces[ o2 ].set( { n3, o3 } );
        n3->next_in_faces[ o3 ].set( { n0, o0 } );
    };

    add_face( n0, n2, n6, n4 );
    add_face( n0, n1, n3, n2 );
    add_face( n1, n5, n7, n3 );
    add_face( n4, n6, n7, n5 );
    add_face( n0, n4, n5, n1 );
    add_face( n2, n3, n7, n6 );
}

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3() {
    nodes_size = 0;
    nodes_rese = block_size;
    nodes = new ( aligned_malloc( nodes_rese / block_size * sizeof( Node ), 64 ) ) Node;

    sphere_radius = 0;
    num_cut_proc  = 0;
}

template<class Pc>
ConvexPolyhedron3<Pc>::~ConvexPolyhedron3() {
    if ( nodes_rese )
        aligned_free( nodes );
}

template<class Pc>
ConvexPolyhedron3<Pc> &ConvexPolyhedron3<Pc>::operator=( const ConvexPolyhedron3 &that ) {
    TODO;
    //    //    this->nb_changes = that.nb_changes;
    //    set_nb_nodes( that.nb_nodes() );

    //    for( std::size_t i = 0; ; i += block_size ) {
    //        Node &n = nodes[ i / block_size ];
    //        const Node &t = that.nodes[ i / block_size ];
    //        if ( i + block_size > nodes_size ) {
    //            std::size_t r = nodes_size - i;
    //            std::memcpy( &n.x, &t.x, r * sizeof( TF ) );
    //            std::memcpy( &n.y, &t.y, r * sizeof( TF ) );
    //            if ( store_the_normals ) {
    //                std::memcpy( &n.dir_x, &t.dir_x, r * sizeof( TF ) );
    //                std::memcpy( &n.dir_y, &t.dir_y, r * sizeof( TF ) );
    //            }
    //            if ( allow_ball_cut ) {
    //                std::memcpy( &n.arc_radius  , &t.arc_radius  , r * sizeof( TF ) );
    //                std::memcpy( &n.arc_center_x, &t.arc_center_x, r * sizeof( TF ) );
    //                std::memcpy( &n.arc_center_y, &t.arc_center_y, r * sizeof( TF ) );
    //            }
    //            for( std::size_t j = 0; j < r; ++j )
    //                n.local_at( j ).cut_id.set( t.local_at( j ).cut_id.get() );
    //            break;
    //        }

    //        std::memcpy( &n.x, &t.x, block_size * sizeof( TF ) );
    //        std::memcpy( &n.y, &t.y, block_size * sizeof( TF ) );
    //        if ( store_the_normals ) {
    //            std::memcpy( &n.dir_x, &t.dir_x, block_size * sizeof( TF ) );
    //            std::memcpy( &n.dir_y, &t.dir_y, block_size * sizeof( TF ) );
    //        }
    //        if ( allow_ball_cut ) {
    //            std::memcpy( &n.arc_radius  , &t.arc_radius  , block_size * sizeof( TF ) );
    //            std::memcpy( &n.arc_center_x, &t.arc_center_x, block_size * sizeof( TF ) );
    //            std::memcpy( &n.arc_center_y, &t.arc_center_y, block_size * sizeof( TF ) );
    //        }
    //        for( std::size_t j = 0; j < block_size; ++j )
    //            n.local_at( j ).cut_id.set( t.local_at( j ).cut_id.get() );
    //    }

    //    sphere_radius = that.sphere_radius;
    //    sphere_center = that.sphere_center;
    //    sphere_cut_id     = that.sphere_cut_id;

    return *this;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::write_to_stream( std::ostream &os ) const {
    TODO;
    //    os << "pos: ";
    //    for( TI i = 0; i < nb_nodes(); ++i )
    //        os << ( i ? " [" : "[" ) << node( i ).pos().x << " " << node( i ).pos().y << "]";
    //    if ( store_the_normals ) {
    //        os << " nrms: ";
    //        for( TI i = 0; i < nb_nodes(); ++i )
    //            os << ( i ? " [" : "[" ) << node( i ).dir() << "]";
    //    }
    //    os << " sphere center: " << sphere_center << " sphere radius: " << sphere_radius;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::display( VtkOutput &vo, const std::vector<TF> &cell_values, Pt /*offset*/, bool display_both_sides ) const {
    std::vector<VtkOutput::Pt> pts;
    faces.foreach( [&]( Face &face ) {
        if ( face.round ) {
            TODO;
        } else if ( display_both_sides || face.cut_id > sphere_cut_id ) {
            pts.clear();
            face.foreach_node( [&]( const Node &node ) {
                pts.push_back( node.pos() );
            } );
            vo.add_polygon( pts, cell_values );
        }
    } );
}

template<class Pc> template<class F>
void ConvexPolyhedron3<Pc>::for_each_edge( const F &f ) const {
    static_assert ( sizeof( Node ) % sizeof( TF ) == 0, "" );
    TODO;

    //    Edge *ptr = edges;
    //    for( TI i = 0; ; ++i ) {
    //        if ( i + block_size >= edges_size ) {
    //            for( TI j = 0; j < edges_size - i; ++j )
    //                f( ptr->local_at( j ) );
    //            break;
    //        }

    //        for( TI j = 0; j < block_size; ++j )
    //            f( ptr->local_at( j ) );
    //        i += block_size;
    //        ++ptr;
    //    }
}

template<class Pc> template<class F>
void ConvexPolyhedron3<Pc>::for_each_node( const F &f ) const {
    static_assert ( sizeof( Node ) % sizeof( TF ) == 0, "" );

    TODO;

    //    Node *ptr = nodes;
    //    for( TI i = 0; ; ++i ) {
    //        if ( i + block_size >= nodes_size ) {
    //            for( TI j = 0; j < nodes_size - i; ++j )
    //                f( ptr->local_at( j ) );
    //            break;
    //        }

    //        for( TI j = 0; j < block_size; ++j )
    //            f( ptr->local_at( j ) );
    //        i += block_size;
    //        ++ptr;
    //    }
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TI ConvexPolyhedron3<Pc>::nb_nodes() const {
    return nodes_size;
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
void ConvexPolyhedron3<Pc>::rese_nb_nodes( TI new_size ) {
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
}

template<class Pc>
void ConvexPolyhedron3<Pc>::set_nb_nodes( TI new_size ) {
    rese_nb_nodes( new_size );
    nodes_size = new_size;
}

template<class Pc> template<int flags>
void ConvexPolyhedron3<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ) {
    TODO;
    //    for( std::size_t num_cut = 0; num_cut < nb_cuts; ++num_cut ) {
    //        // distances
    //        TI nb_outside_nodes = 0;
    //        TF rd = cut_ps[ num_cut ];
    //        TF nx = cut_dir[ 0 ][ num_cut ];
    //        TF ny = cut_dir[ 1 ][ num_cut ];
    //        TF nz = cut_dir[ 2 ][ num_cut ];
    //        #ifdef __AVX5d12F__
    //        //        __m512d rd = _mm512_set1_pd( cut_ps[ num_cut ] );
    //        //        __m512d nx = _mm512_set1_pd( cut_dir[ 0 ][ num_cut ] );
    //        //        __m512d ny = _mm512_set1_pd( cut_dir[ 1 ][ num_cut ] );
    //        //        __m512d nz = _mm512_set1_pd( cut_dir[ 2 ][ num_cut ] );
    //        //        for( std::size_t n = 0; n < nb_nodes(); n += 8 ) {
    //        //            __m512d px_0 = _mm512_load_pd( x + 0 );
    //        //            __m512d py_0 = _mm512_load_pd( y + 0 );
    //        //            __m512i pc_0 = _mm512_load_epi64( c + 0 );
    //        //            __m512d bi_0 = _mm512_add_pd( _mm512_mul_pd( px_0, nx ), _mm512_mul_pd( py_0, ny ) );
    //        //            std::uint8_t outside_0 = _mm512_cmp_pd_mask( bi_0, rd, _CMP_GT_OQ );
    //        //            __m512d di_0 = _mm512_sub_pd( bi_0, rd );
    //        //        }
    //        TODO;
    //        #else
    //        for_each_node( [&]( Node &node ) {
    //            node.d = node.x * nx + node.y * ny + node.z * nz - rd;
    //            nb_outside_nodes += node.outside();
    //        } );
    //        if ( nb_outside_nodes == 0 )
    //            return;
    //        if ( nb_outside_nodes == nb_nodes() ) {
    //            nodes_size = 0;
    //            return;
    //        }
    //        #endif

    //        // reservation for the new nodes and the new edges. Make a linked list of face impacted by the cut
    //        ++num_cut_proc;
    //        int last_cut_face = -1;
    //        TI nb_partially_outside_edges = 0;
    //        for_each_edge( [&]( Edge &edge ) {
    //            bool o0 = node( edge.node_0 ).outside();
    //            bool o1 = node( edge.node_1 ).outside();
    //            nb_partially_outside_edges += o0 ^ o1;

    //            if ( o0 || o1 ) {
    //                auto reg_face = [&]( int num_face ) {
    //                    Face &face = faces[ num_face ];
    //                    if ( face.num_cut_proc != num_cut_proc ) {
    //                        face.num_cut_proc = num_cut_proc;
    //                        face.prev_cut_face = last_cut_face;
    //                        last_cut_face = num_face;
    //                    }
    //                };
    //                reg_face( edge.face_0 );
    //                reg_face( edge.face_1 );
    //            }
    //        } );

    //        TI old_edges_size = edges_size;
    //        TI old_nodes_size = nodes_size;
    //        rese_nb_edges( edges_size + nb_partially_outside_edges );
    //        rese_nb_nodes( nodes_size + nb_partially_outside_edges );

    //        // make the new nodes and adjust the old edges.
    //        for_each_edge( [&]( Edge &edge ) {
    //            bool o0 = node( edge.node_0 ).outside();
    //            bool o1 = node( edge.node_1 ).outside();
    //            if ( o0 ) {
    //                if ( ! o1 ) {
    //                    node( nodes_size ).set_pos( node( edge.node_0 ).pos() + node( edge.node_0 ).d / ( node( edge.node_0 ).d - node( edge.node_1 ).d ) * ( node( edge.node_1 ).pos() - node( edge.node_0 ).pos() ) );
    //                    edge.node_0 = nodes_size++;
    //                }
    //            } else if ( o1 ) {
    //                node( nodes_size ).set_pos( node( edge.node_0 ).pos() + node( edge.node_0 ).d / ( node( edge.node_0 ).d - node( edge.node_1 ).d ) * ( node( edge.node_1 ).pos() - node( edge.node_0 ).pos() ) );
    //                edge.node_1 = nodes_size++;
    //            }
    //        } );

    //        // mark faces to be removed
    //        ++num_cut_proc;
    //        int last_rem_face = -1;
    //        for( int cut_face = last_cut_face; cut_face >= 0; ) {
    //            Face &face = faces[ cut_face ];
    //            int prev_cut_face = face.prev_cut_face;

    //            int nb_out_points = 0;
    //            for( TI i = face.num_in_edge_beg; i < face.num_in_edge_end; ++i ) {
    //                int n0 = edge_n0( num_in_edges_m2[ i ] );
    //                if ( node( n0 ).outside() ) {
    //                    ++nb_out_points;
    //                }
    //            }

    //            if ( nb_out_points == face.num_in_edge_end - face.num_in_edge_beg ) {
    //                face.num_cut_proc = num_cut_proc;
    //                face.prev_cut_face = last_rem_face;
    //                last_rem_face = cut_face;
    //            }

    //            cut_face = prev_cut_face;
    //        }

    //        // remove void faces
    //        int last_valid_face = faces.size() - 1;
    //        for( int cut_face = last_rem_face; cut_face >= 0; cut_face = faces[ cut_face ].prev_cut_face ) {
    //            while ( last_valid_face >= 0 && faces[ last_valid_face ].num_cut_proc == num_cut_proc )
    //                --last_valid_face;

    //            if ( last_valid_face > cut_face ) {
    //                TODO;
    //            }

    //            faces.pop_back();
    //            P( cut_face );
    //        }
    //    }
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
