//#include "../Support/Display/generic_ostream_output.h"
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
#include <map>
#include <set>

//#define _USE_MATH_DEFINES
//#include <math.h>

// #include "Internal/(convex_polyhedron_plane_cut_simd_switch.cpp).h"

namespace sdot {

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3( const Box &box ) : ConvexPolyhedron3() {
    set_box( box );
}


template<class Pc>
void ConvexPolyhedron3<Pc>::set_box( const Box &box ) {
    // nodes
    set_nb_nodes( 8 );

    auto set_node = [&]( TI index, TF x, TF y, TF z ) -> Node * {
        Node *res = &node( index );
        for( TI i = 0; i < 3; ++i )
            res->next_in_faces[ i ].set( { nullptr, 0 } );
        res->repl.set( res );
        res->x = x;
        res->y = y;
        res->z = z;

        return res;
    };

    Node *n[ 8 ] = {
        set_node( 0, box.p0.x, box.p0.y, box.p0.z ),
        set_node( 1, box.p1.x, box.p0.y, box.p0.z ),
        set_node( 2, box.p0.x, box.p1.y, box.p0.z ),
        set_node( 3, box.p1.x, box.p1.y, box.p0.z ),
        set_node( 4, box.p0.x, box.p0.y, box.p1.z ),
        set_node( 5, box.p1.x, box.p0.y, box.p1.z ),
        set_node( 6, box.p0.x, box.p1.y, box.p1.z ),
        set_node( 7, box.p1.x, box.p1.y, box.p1.z )
    };

    // faces
    auto add_face = [&]( int n0, int n1, int n2, int n3, int off, Pt normal, int se0, int se1, int se2, int se3 ) {
        Face *face = faces.create();
        if ( allow_ball_cut )
            face->round = false;
        face->num_cut_proc = 0;
        face->normal = normal;
        face->cut_id = box.cut_id;

        n[ n0 ]->next_in_faces[ off ].set( { n[ n1 ], off } );
        n[ n1 ]->next_in_faces[ off ].set( { n[ n2 ], off } );
        n[ n2 ]->next_in_faces[ off ].set( { n[ n3 ], off } );
        n[ n3 ]->next_in_faces[ off ].set( { n[ n0 ], off } );

        face->first_edge = { n[ n0 ], off };

        n[ n0 ]->faces[ off ].set( face );
        n[ n1 ]->faces[ off ].set( face );
        n[ n2 ]->faces[ off ].set( face );
        n[ n3 ]->faces[ off ].set( face );

        n[ n0 ]->sibling_edges[ off ].set( { n[ n1 ], se0 } );
        n[ n1 ]->sibling_edges[ off ].set( { n[ n2 ], se1 } );
        n[ n2 ]->sibling_edges[ off ].set( { n[ n3 ], se2 } );
        n[ n3 ]->sibling_edges[ off ].set( { n[ n0 ], se3 } );
    };

    add_face( 0, 2, 3, 1,  0,  { 0, 0, -1 },  2, 1, 2, 1 ); // count=11110000
    add_face( 4, 5, 7, 6,  0,  { 0, 0, +1 },  1, 2, 1, 2 ); // count=11111111
    add_face( 0, 1, 5, 4,  1,  { 0, -1, 0 },  0, 2, 0, 2 ); // count=22112211
    add_face( 2, 6, 7, 3,  1,  { 0, +1, 0 },  2, 0, 2, 0 ); // count=22222222
    add_face( 0, 4, 6, 2,  2,  { -1, 0, 0 },  1, 0, 1, 0 ); // ...
    add_face( 1, 3, 7, 5,  2,  { +1, 0, 0 },  0, 1, 0, 1 ); //
}

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3() {
    nodes_size = 0;
    nodes_rese = 4 * block_size;
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
ConvexPolyhedron3<Pc> &ConvexPolyhedron3<Pc>::operator=( const ConvexPolyhedron3 &/*that*/ ) {
    //    sphere_radius = that.sphere_radius;
    //    sphere_center = that.sphere_center;
    //    sphere_cut_id = that.sphere_cut_id;
    //    num_cut_proc  = that.num_cut_proc;

    //    set_nb_nodes( that.nb_nodes() );

    //    // make the new nodes
    //    for( std::size_t i = 0; ; i += block_size ) {
    //        Node &n = nodes[ i / block_size ];
    //        const Node &t = that.nodes[ i / block_size ];
    //        if ( i + block_size > nodes_size ) {
    //            std::size_t r = nodes_size - i;
    //            std::memcpy( &n.x, &t.x, r * sizeof( TF ) );
    //            std::memcpy( &n.y, &t.y, r * sizeof( TF ) );
    //            std::memcpy( &n.z, &t.z, r * sizeof( TF ) );
    //            if ( store_the_normals ) {
    //                std::memcpy( &n.dir_x, &t.dir_x, r * sizeof( TF ) );
    //                std::memcpy( &n.dir_y, &t.dir_y, r * sizeof( TF ) );
    //                std::memcpy( &n.dir_z, &t.dir_z, r * sizeof( TF ) );
    //            }
    //            if ( allow_ball_cut ) {
    //                TODO;
    //            }
    //            for( std::size_t j = 0; j < r; ++j ) {
    //                const Node &nt = t.local_at( j );
    //                Node &nn = n.local_at( j );

    //                nn.cut_id.set( nt.cut_id.get() );

    //                for( std::size_t k = 0; k < 3; ++k ) {
    //                    nn.next_in_faces[ k ].set( t.next_in_faces[ k ].get().displaced( nodes, that.nodes ) );
    //                    nn.sibling_edges[ k ].set( t.next_in_faces[ k ].get().displaced( nodes, that.nodes ) );
    //                    nn.faces[ k ].set( t.faces[ k ].get().displaced( nodes, that.nodes ) );
    //                }
    //            }
    //            break;
    //        }

    //        TODO;
    //        //        std::memcpy( &n.x, &t.x, block_size * sizeof( TF ) );
    //        //        std::memcpy( &n.y, &t.y, block_size * sizeof( TF ) );
    //        //        std::memcpy( &n.z, &t.z, block_size * sizeof( TF ) );
    //        //        if ( store_the_normals ) {
    //        //            std::memcpy( &n.dir_x, &t.dir_x, block_size * sizeof( TF ) );
    //        //            std::memcpy( &n.dir_y, &t.dir_y, block_size * sizeof( TF ) );
    //        //            std::memcpy( &n.dir_z, &t.dir_z, block_size * sizeof( TF ) );
    //        //        }
    //        //        if ( allow_ball_cut ) {
    //        //            TODO;
    //        //        }
    //        //        for( std::size_t j = 0; j < block_size; ++j )
    //        //            n.local_at( j ).cut_id.set( t.local_at( j ).cut_id.get() );

    //        //        PNoI        next_in_faces[ 3 ];      ///< for each edge, address + offset (between 0 and 3) in the `_in_faces` lists
    //        //        PNoI        sibling_edges[ 3 ];      ///< for each edge, address + offset for sibling edges
    //        //        PFace       faces[ 3 ];              ///< for each edge
    //    }

    //    // make the new faces
    //    //    faces.clear();
    //    //    that.for_each_face( [&]( const Face &f ) {
    //    //        Face *nf = faces.create();
    //    //    } );
    TODO;

    return *this;
}

template<class Pc>
ConvexPolyhedron3<Pc> &ConvexPolyhedron3<Pc>::operator=( const Box &box ) {
    sphere_radius = -1;
    faces.clear();

    set_box( box );

    return *this;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::write_to_stream( std::ostream &os, bool /*debug*/ ) const {
    std::map<const Node *,std::string> node_map;
    std::map<const Face *,std::string> face_map;

    //    for_each_edge( [&]( const Edge &edge ) {} );
    for_each_node( [&]( const Node &node ) { node_map[ &node ] = node_map.size() < 26 ? 'A' + node_map.size() : 'a' + node_map.size() - 26; } );
    for_each_face( [&]( const Face &face ) { face_map[ &face ] = 'a' + face_map.size(); } );

    os << "Nodes:\n";
    for_each_node( [&]( const Node &node ) {
        node.pos().write_to_stream( os << "  " << node_map[ &node ] << " => " );
        os << "\n";
    } );

    os << "Faces:\n";
    for_each_face( [&]( const Face &face ) {
        os << "  " << face_map[ &face ] << " => d=";
        face.normal.write_to_stream( os );

        os << " n=[";
        face.foreach_node( [&]( const Node &node ) {
            os << node_map[ &node ];
        } );
        os << "]";
        os << "\n";

        face.foreach_edge( [&]( const Edge &edge ) {
            os << "    " << node_map[ edge.n0() ] << " -> " << node_map[ edge.n1() ];
            os << " f:" << face_map[ edge.face() ];
            os << " s:" << node_map[ edge.sibling().n0() ] << node_map[ edge.sibling().n1() ];
            os << " fs:" << face_map[ edge.sibling().face() ];
            os << "\n";
        } );
    } );
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
void ConvexPolyhedron3<Pc>::display_vtk( VtkOutput &vo, const std::vector<TF> &cell_values, Pt offset, bool /*display_both_sides*/ ) const {
    std::vector<VtkOutput::Pt> pts;
    for_each_face( [&]( const Face &face ) {
        if ( allow_ball_cut && face.round ) {
            TODO;
        } else /*if ( display_both_sides || face.cut_id > sphere_cut_id )*/ {
            pts.clear();
            face.foreach_node( [&]( const Node &node ) {
                pts.push_back( node.pos() + offset );
            } );
            vo.add_polygon( pts, cell_values );
        }
    } );
}

template<class Pc> template<class F>
void ConvexPolyhedron3<Pc>::for_each_edge( const F &/*f*/ ) const {
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
void ConvexPolyhedron3<Pc>::for_each_face( const F &f ) const {
    faces.foreach( f );
}

template<class Pc> template<class F>
void ConvexPolyhedron3<Pc>::for_each_node( const F &f ) const {
    static_assert ( sizeof( Node ) % sizeof( TF ) == 0, "" );

    Node *ptr = nodes;
    for( TI i = 0, s = nodes_size; ; ) {
        if ( i + block_size >= s ) {
            for( TI j = 0; j < s - i; ++j )
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

        TODO; // pointers
    }
}

template<class Pc>
void ConvexPolyhedron3<Pc>::set_nb_nodes( TI new_size ) {
    rese_nb_nodes( new_size );
    nodes_size = new_size;
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::Node *ConvexPolyhedron3<Pc>::new_node( Pt pos ) {
    TI n = nb_nodes();
    set_nb_nodes( n + 1 );

    Node *res = &node( n );
    res->repl.set( res );
    res->set_pos( pos );
    return res;
}


template<class Pc> template<int flags>
void ConvexPolyhedron3<Pc>::plane_cut_mt_64( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ) {
    // we assume here that cells with nb nodes > 64 are not common. Thus this procedure is no fully optimized.
    for( std::size_t num_cut = 0; num_cut < nb_cuts; ++num_cut ) {
        TF nx = cut_dir[ 0 ][ num_cut ];
        TF ny = cut_dir[ 1 ][ num_cut ];
        TF nz = cut_dir[ 2 ][ num_cut ];
        TF rd = cut_ps[ num_cut ];

        // get distances + nb_outside_nodes
        TI nb_outside_nodes = 0;
        for_each_node( [&]( Node &node ) {
            node.d = node.x * nx + node.y * ny + node.z * nz - rd;
            nb_outside_nodes += node.outside();
        } );

        if ( nb_outside_nodes == 0 )
            continue;
        if ( nb_outside_nodes == nb_nodes() ) {
            for_each_face( [&]( Face &face ) { faces.free( &face ); } );
            nodes_size = 0;
            return;
        }

        // find the faces with at least one outside node
        ++num_cut_proc;
        Node *last_created_node, *last_free_node = nullptr, *first_free_node;
        for_each_node( [&]( Node &node ) {
            if ( node.outside() == false )
                return;

            if ( last_free_node )
                last_free_node->repl.set( &node );
            else
                first_free_node = &node;
            last_free_node = &node;

            for( std::size_t i = 0; i < 3; ++i ) {
                Face *face = node.faces[ i ].get();
                if ( face->num_cut_proc == num_cut_proc )
                    continue;
                face->num_cut_proc = num_cut_proc;

                // helpers to find a inside <-> outside edge
                auto find_outside_inside = [&]( Edge beg, Node *end ) -> Edge {
                    while ( true ) {
                        if ( beg.n0()->outside() && beg.n1()->outside() == false )
                            return beg;
                        if ( beg.n1() == end )
                            return nullptr;
                        beg = beg.next();
                    }
                };
                auto find_inside_outside = [&]( Edge beg ) -> Edge {
                    while ( true ) {
                        if ( beg.n0()->outside() == false && beg.n1()->outside() )
                            return beg;
                        beg = beg.next();
                    }
                };

                // find a first inside -> outside edge
                Edge oie = find_outside_inside( face->first_edge, face->first_edge.n0() );
                if ( ! oie ) {
                    faces.free( face );
                    continue;
                }

                // get outside -> inside edge, siblings + register
                Edge ioe = find_inside_outside( oie.next() );
                Edge se_oie = oie.sibling();
                Edge se_ioe = ioe.sibling();
                face->first_edge = ioe;

                // creation/retrieval of the new node in oie
                Node *n_oie;
                int num_in_oie;
                if ( se_oie.face()->num_cut_proc == num_cut_proc ) {
                    n_oie = se_oie.n1();

                    se_oie.n0()->sibling_edges[ se_oie.offset() ].set( { n_oie, 1 } );
                    n_oie->sibling_edges[ 1 ].set( se_oie );

                    n_oie->next_in_faces[ 1 ].set( oie.next() );
                    n_oie->faces[ 1 ].set( face );
                    num_in_oie = 1;
                } else {
                    Pt p = oie.n0()->pos() + oie.n0()->d / ( oie.n0()->d - oie.n1()->d ) * ( oie.n1()->pos() - oie.n0()->pos() );
                    n_oie = new_node( p );
                    n_oie->d = -1;

                    n_oie->next_in_faces[ 0 ].set( oie.next() );
                    n_oie->faces[ 0 ].set( face );
                    num_in_oie = 0;

                    oie.n0()->next_in_faces[ oie.offset() ].set( Edge( n_oie, 0 ) ); // to have n_oie in se_ioe.n1() in the next round
                }

                // creation/retrieval of the new node in ioe (next outside -> inside edge)
                if ( se_ioe.face()->num_cut_proc == num_cut_proc ) {
                    Node *n_ioe = se_ioe.n1();

                    ioe.n0()->sibling_edges[ ioe.offset() ].set( { n_ioe, 0 } );
                    n_ioe->sibling_edges[ 0 ].set( ioe );

                    ioe.n0()->next_in_faces[ ioe.offset() ].set( Edge( n_ioe, 1 ) );
                    n_ioe->next_in_faces[ 1 ].set( Edge( n_oie, num_in_oie ) );
                    n_ioe->faces[ 1 ].set( face );
                } else {
                    Pt p = ioe.n0()->pos() + ioe.n0()->d / ( ioe.n0()->d - ioe.n1()->d ) * ( ioe.n1()->pos() - ioe.n0()->pos() );
                    Node *n_ioe = new_node( p );
                    n_ioe->d = -1;

                    ioe.n0()->next_in_faces[ ioe.offset() ].set( Edge( n_ioe, 0 ) );
                    n_ioe->next_in_faces[ 0 ].set( Edge( n_oie, num_in_oie ) );
                    n_ioe->faces[ 0 ].set( face );
                    last_created_node = n_ioe;
                }
            }
        } );

        // creation of the new face
        Face *face = faces.create();
        face->normal = { nx, ny, nz };
        face->cut_id = cut_id[ num_cut ];
        face->num_cut_proc = 0;
        face->first_edge = { last_created_node, 2 };
        for( Edge e( last_created_node, 0 ); ; e = e.next().with_1_xored_offset() ) { // offset = 0 at the beginning because last_created_node = n_ioe
            Node *n1 = e.n1();

            e.n0()->sibling_edges[ e.offset() ].set( { n1, 2 } );
            n1->sibling_edges[ 2 ].set( e );

            n1->next_in_faces[ 2 ].set( { e.n0(), 2 } );
            n1->faces[ 2 ].set( face );

            if ( n1 == last_created_node )
                break;
        }

        // get node replacement locations + mark faces with replaced nodes
        ++num_cut_proc;
        Face *last_marked_face = nullptr;
        last_free_node->repl.set( nullptr );
        std::size_t num_node_to_move = nb_nodes();
        for( Node *node = first_free_node; ; ) {
            // find the node to be moved to `node`
            Node *n = &this->node( --num_node_to_move );
            if ( n <= node )
                break;
            if ( n->inside() ) {
                for( std::size_t i = 0; i < 3; ++i ) {
                    Face *face = n->faces[ i ].get();
                    if ( face->num_cut_proc == num_cut_proc )
                        continue;
                    face->num_cut_proc = num_cut_proc;

                    face->prev_marked = last_marked_face;
                    last_marked_face = face;
                }

                node->get_straight_content_from( *n );
                n->repl.set( node );

                node = node->repl.get();
                if ( ! node )
                    break;
            }
        }

        // modify node pointers in edges
        for( Face *face = last_marked_face; face; face = face->prev_marked ) {
            Edge e = face->first_edge;
            for( Node *n0 = e.n0(); ; ) {
                Node *node = e.n0(), *repl = node->repl.get();
                int offset = e.offset();
                Edge ne = e.next();

                repl->next_in_faces[ offset ].set( node->next_in_faces[ offset ].get().repl() );
                repl->sibling_edges[ offset ].set( node->sibling_edges[ offset ].get().repl() );

                if ( ne.n0() == n0 )
                    break;
                e = ne;
            }

            face->first_edge = e.repl();
        }

        // set node->repl to node
        for( Node *node = first_free_node; ; ) {
            Node *next = node->repl.get();
            node->repl.set( node );

            if ( ! next )
                break;
            node = next;
        }

        // modify nb nodes
        nodes_size -= nb_outside_nodes;
    }
}

template<class Pc> template<int flags>
void ConvexPolyhedron3<Pc>::plane_cut_lt_64( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ) {
    #ifdef __AVX5d12F__
    //        __m512d rd = _mm512_set1_pd( cut_ps[ num_cut ] );
    //        __m512d nx = _mm512_set1_pd( cut_dir[ 0 ][ num_cut ] );
    //        __m512d ny = _mm512_set1_pd( cut_dir[ 1 ][ num_cut ] );
    //        __m512d nz = _mm512_set1_pd( cut_dir[ 2 ][ num_cut ] );
    //        for( std::size_t n = 0; n < nb_nodes(); n += 8 ) {
    //            __m512d px_0 = _mm512_load_pd( x + 0 );
    //            __m512d py_0 = _mm512_load_pd( y + 0 );
    //            __m512i pc_0 = _mm512_load_epi64( c + 0 );
    //            __m512d bi_0 = _mm512_add_pd( _mm512_mul_pd( px_0, nx ), _mm512_mul_pd( py_0, ny ) );
    //            std::uint8_t outside_0 = _mm512_cmp_pd_mask( bi_0, rd, _CMP_GT_OQ );
    //            __m512d di_0 = _mm512_sub_pd( bi_0, rd );
    //        }
    #else
    #endif

    return plane_cut_mt_64( cut_dir, cut_ps, cut_id, nb_cuts, N<flags>() );
}

template<class Pc> template<int flags>
void ConvexPolyhedron3<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ) {
    if ( nb_nodes() <= 64 )
        return plane_cut_lt_64( cut_dir, cut_ps, cut_id, nb_cuts, N<flags>() );
    return plane_cut_mt_64( cut_dir, cut_ps, cut_id, nb_cuts, N<flags>() );

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
}

template<class Pc>
void ConvexPolyhedron3<Pc>::check() const {
    std::set<const Node *> used_nodes;
    for_each_face( [&]( const Face &face ) {
        face.foreach_node( [&]( const Node &node ) {
            used_nodes.insert( &node );
        } );

        face.foreach_edge( [&]( Edge edge ) {
            if ( edge.face() != &face )
                P( "bad face" );

            Edge sedge = edge.sibling();
            if ( edge.n1() != sedge.n0() || edge.n0() != sedge.n1() )
                P( "bad sibling" );

            Edge ssedge = sedge.sibling();
            if ( edge.n0() != ssedge.n0() || edge.n1() != ssedge.n1() )
                P( "bad sibling 2" );
            if ( edge.face() != ssedge.face() )
                P( "bad sibling 3" );
        } );
    } );

    //    for_each_node( [&]( const Node &node ) {
    //        if ( used_nodes.count( &node ) == 0 )
    //            P( "lonely node" );
    //    } );
}


template<class Pc>
void ConvexPolyhedron3<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ) {
    return plane_cut( cut_dir, cut_ps, cut_id, nb_cuts, N<0>() );
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::integral() const {
    TF res;
    if ( faces.empty() ) {
        res = sphere_radius > 0 ? 4 * M_PI / 3 * std::pow( sphere_radius, 3 ) : 0;
    } else {
        res = 0;
        faces.foreach( [&]( const Face &fp ) {
            if ( allow_ball_cut && fp.round )
                TODO;
            res += dot( fp.first_node()->pos(), fp.normal ) * fp.flat_area() / 3;
        } );

        //        if ( round_surfaces.size() == 1 ) {
        //            res = sphere_radius * area( round_surfaces[ 0 ] ) / 3;
        //        } else {
        //            // we substract area of the hole from area of the full sphere
        //            TF sa = 4 * M_PI * std::pow( sphere_radius, 2 );
        //            TF pa = sa * ( TF( 1 ) - nb_connections );
        //            for( const RoundSurface &rp : round_surfaces )
        //                pa += area( rp );
        //            res = sphere_radius * pa / 3;
        //        }

        //        const Pt &sc = sphere_center;
        //        for( const FlatSurface &fp : flat_surfaces )
        //            res += dot( cut_info[ fp.cut_index ].cut_O - sc, cut_info[ fp.cut_index ].cut_N ) * area( fp ) / 3;
    }

    //    for( const Hole &hole : holes ) {
    //        TF h = sphere_radius - dot( hole.cut_O - sphere_center, hole.cut_N );
    //        res -= M_PI / 3 * h * h * ( 3 * sphere_radius - h );
    //    }

    return res;
}

template<class Pc> template<class TL>
void ConvexPolyhedron3<Pc>::BoundaryItem::add_simplex_list( TL &/*lst*/ ) const {
    // lst.push_back( { points[ 0 ], points[ 1 ] } );
}

} // namespace sdot
