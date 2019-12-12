//#include "../Support/Display/generic_ostream_output.h"
#include "../Support/Display/binary_repr.h"
#include "../Support/aligned_memory.h"
#include "../Support/bit_handling.h"
#include "../Support/SimdRange.h"
#include "../Support/SimdVec.h"
#include "../Support/ASSERT.h"
#include "../Support/TODO.h"
#include "../Support/pi.h"
#include "../Support/P.h"
#include "ConvexPolyhedron3b.h"
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
ConvexPolyhedron3<Pc>::Box::Box( Pt p0, Pt p1, CI cut_id ) : cp( p0, p1, cut_id ) {
}

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3( Pt p0, Pt p1, CI cut_id ) : ConvexPolyhedron3() {
    // sizes
    nodes_size = 8;
    faces_size = 6;

    // nodes
    auto set_node = [&]( int index, TF x, TF y, TF z ) -> Node * {
        Node &res = nodes.local_at( index );
        res.x = x;
        res.y = y;
        res.z = z;

        return &res;
    };

    Node *n[ 8 ] = {
        set_node( 0, p0.x, p0.y, p0.z ),
        set_node( 1, p1.x, p0.y, p0.z ),
        set_node( 2, p0.x, p1.y, p0.z ),
        set_node( 3, p1.x, p1.y, p0.z ),
        set_node( 4, p0.x, p0.y, p1.z ),
        set_node( 5, p1.x, p0.y, p1.z ),
        set_node( 6, p0.x, p1.y, p1.z ),
        set_node( 7, p1.x, p1.y, p1.z )
    };

    // faces
    auto add_face = [&]( TI index, int n0, int n1, int n2, int n3, Pt normal ) {
        constexpr std::uint64_t m = 1;

        faces.node_masks[ index ] = ( m << n0 ) | ( m << n1 ) | ( m << n2 ) | ( m << n3 ) ;
        faces.normal_xs [ index ] = normal.x;
        faces.normal_ys [ index ] = normal.y;
        faces.normal_zs [ index ] = normal.z;
        faces.nb_nodes  [ index ] = 4;
        faces.cut_ids   [ index ] = cut_id;

        auto &ln = faces.node_lists[ index ];
        ln[ 0 ] = n0;
        ln[ 1 ] = n1;
        ln[ 2 ] = n2;
        ln[ 3 ] = n3;
    };

    add_face( 0,  0, 2, 3, 1,  { 0, 0, -1 } );
    add_face( 1,  4, 5, 7, 6,  { 0, 0, +1 } );
    add_face( 2,  0, 1, 5, 4,  { 0, -1, 0 } );
    add_face( 3,  2, 6, 7, 3,  { 0, +1, 0 } );
    add_face( 4,  0, 4, 6, 2,  { -1, 0, 0 } );
    add_face( 5,  1, 3, 7, 5,  { +1, 0, 0 } );
}

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3() {
    sphere_radius = 0;
    num_cut_proc  = 0;
    nodes_size    = 0;
    faces_size    = 0;

    for( std::size_t i = 0; i < max_nb_edges; ++i )
        edge_num_cut_procs[ i ] = 0;

    // to please valgrind
    for( std::size_t i = 0; i < ConvexPolyhedron3Lt64FaceBlock<Pc>::max_nb_faces_per_cell; ++i )
        for( std::size_t j = 0; j < ConvexPolyhedron3Lt64FaceBlock<Pc>::max_nb_nodes_per_face; ++j )
            faces.node_lists[ i ][ j ] = 0;
}


template<class Pc>
ConvexPolyhedron3<Pc>::~ConvexPolyhedron3() {
}

template<class Pc>
ConvexPolyhedron3<Pc> &ConvexPolyhedron3<Pc>::operator=( const ConvexPolyhedron3 &that ) {
    nodes_size = that.nodes_size;
    faces_size = that.faces_size;

    std::memcpy( &nodes.x, &that.nodes.x, nodes_size * sizeof( TF ) );
    std::memcpy( &nodes.y, &that.nodes.y, nodes_size * sizeof( TF ) );
    std::memcpy( &nodes.z, &that.nodes.z, nodes_size * sizeof( TF ) );

    std::memcpy( faces.node_lists, that.faces.node_lists, faces_size * sizeof( faces.node_lists[ 0 ] ) );
    std::memcpy( faces.node_masks, that.faces.node_masks, faces_size * sizeof( faces.node_masks[ 0 ] ) );
    std::memcpy( faces.normal_xs , that.faces.normal_xs , faces_size * sizeof( faces.normal_xs [ 0 ] ) );
    std::memcpy( faces.normal_ys , that.faces.normal_ys , faces_size * sizeof( faces.normal_xs [ 0 ] ) );
    std::memcpy( faces.normal_zs , that.faces.normal_zs , faces_size * sizeof( faces.normal_xs [ 0 ] ) );
    std::memcpy( faces.nb_nodes  , that.faces.nb_nodes  , faces_size * sizeof( faces.nb_nodes  [ 0 ] ) );
    std::memcpy( faces.cut_ids   , that.faces.cut_ids   , faces_size * sizeof( faces.cut_ids   [ 0 ] ) );

    return *this;
}

template<class Pc>
ConvexPolyhedron3<Pc> &ConvexPolyhedron3<Pc>::operator=( const Box &that ) {
    constexpr unsigned _nodes_size = 8;
    constexpr unsigned _faces_size = 6;

    nodes_size = _nodes_size;
    faces_size = _faces_size;

    std::memcpy( &nodes.x, &that.cp.nodes.x, _nodes_size * sizeof( TF ) );
    std::memcpy( &nodes.y, &that.cp.nodes.y, _nodes_size * sizeof( TF ) );
    std::memcpy( &nodes.z, &that.cp.nodes.z, _nodes_size * sizeof( TF ) );

    std::memcpy( faces.node_lists, that.cp.faces.node_lists, _faces_size * sizeof( faces.node_lists[ 0 ] ) );
    std::memcpy( faces.node_masks, that.cp.faces.node_masks, _faces_size * sizeof( faces.node_masks[ 0 ] ) );
    std::memcpy( faces.normal_xs , that.cp.faces.normal_xs , _faces_size * sizeof( faces.normal_xs [ 0 ] ) );
    std::memcpy( faces.normal_ys , that.cp.faces.normal_ys , _faces_size * sizeof( faces.normal_xs [ 0 ] ) );
    std::memcpy( faces.normal_zs , that.cp.faces.normal_zs , _faces_size * sizeof( faces.normal_xs [ 0 ] ) );
    std::memcpy( faces.nb_nodes  , that.cp.faces.nb_nodes  , _faces_size * sizeof( faces.nb_nodes  [ 0 ] ) );
    std::memcpy( faces.cut_ids   , that.cp.faces.cut_ids   , _faces_size * sizeof( faces.cut_ids   [ 0 ] ) );

    return *this;
}

template<class Pc>
void ConvexPolyhedron3<Pc>::write_to_stream( std::ostream &os, bool /*debug*/ ) const {
    os << "Nodes:";
    for_each_node( [&]( const auto &node ) {
        os << "\n  " << node.x << " " << node.y << " " << node.z;
    } );

    os << "\nFaces:";
    for_each_face( [&]( const auto &face ) {
        os << "\n ";
        face.for_each_node_index( [&]( int index ) {
            os << " " << index;
        } );
        os << " " << binary_repr( faces.node_masks[ face.num_face ] );
    } );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::display_vtk( VtkOutput &vo, const std::vector<TF> &cell_values, Pt offset, bool /*display_both_sides*/ ) const {
    std::vector<VtkOutput::Pt> pts;
    pts.reserve( 16 );
    for_each_face( [&]( const Face &face ) {
        if ( allow_ball_cut ) { //  && face.round
            TODO;
        } else /*if ( display_both_sides || face.cut_id > sphere_cut_id )*/ {
            pts.resize( 0 );
            face.for_each_node( [&]( const Node &node ) {
                pts.push_back( node.pos() + offset );
            } );
            vo.add_polygon( pts, cell_values );
        }
    } );
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
    return nodes.local_at( index );
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::Node &ConvexPolyhedron3<Pc>::node( TI index ) {
    return nodes.local_at( index );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::for_each_boundary_item( const std::function<void( const BoundaryItem &boundary_item )> &f ) const {
    for_each_face( f );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::for_each_face( const std::function<void( const Face & )> &f ) const {
    for( int num_face = 0; num_face < faces_size; ++num_face )
        f( { num_face, this } );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::for_each_node( const std::function<void( const Node & )> &f ) const {
    for( int num_node = 0; num_node < nodes_size; ++num_node )
        f( nodes.local_at( num_node ) );
}


template<class Pc> template<int flags>
std::size_t ConvexPolyhedron3<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_ids, std::size_t nb_cuts, N<flags> ) {
    constexpr int ss = SimdSize<TF>::value;
    additional_nodes.clear();
    additional_nums.clear();

    for( std::size_t num_cut = 0; num_cut < nb_cuts; ++num_cut ) {
        // get distance and outside bit for each node
        TF cx = cut_dir[ 0 ][ num_cut ];
        TF cy = cut_dir[ 1 ][ num_cut ];
        TF cz = cut_dir[ 2 ][ num_cut ];
        TF cs = cut_ps[ num_cut ];
        std::uint64_t outside_nodes = 0;
        SimdRange<ss>::for_each( nodes_size, [&]( int n, auto s ) {
            using LF = SimdVec<TF,s.val>;
            LF px = LF::load_aligned( &nodes.x + n );
            LF py = LF::load_aligned( &nodes.y + n );
            LF pz = LF::load_aligned( &nodes.z + n );

            LF bi = px * LF( cx ) + py * LF( cy ) + pz * LF( cz );
            LF::store_aligned( &nodes.d + n, bi - LF( cs ) );
            std::uint64_t lo = bi > LF( cs );
            outside_nodes |= lo << n;
        } );
        
        // if nothing has changed => go to the next cut
        if ( outside_nodes == 0 )
            continue;

        // => we're starting a new cut proc (used e.g. for edge_num_cuts)
        ++num_cut_proc;

        // new nodes that are to be moved to place of old ones
        std::uint8_t repl_node_dsts[ max_nb_nodes ];
        std::uint8_t repl_node_srcs[ max_nb_nodes ];
        int nb_repl_nodes = 0;

        // list of faces to be removes
        int faces_to_rem[ Lt64FaceBlock::max_nb_faces_per_cell ];
        int nb_faces_to_rem = 0;

        // where to put the next temporary node
        int ind_nxt_tmp_node = max_nb_nodes;

        // linked list of nodes for the new face
        std::uint8_t prev_cut_nodes[ max_nb_nodes ]; // index to the next node
        std::uint8_t last_cut_node;

        // copy of `ou` to get usable indices for the new nodes
        std::uint64_t available_nodes = outside_nodes;

        // handle intersected faces
        auto handle_intersected_face = [&]( unsigned num_face ) {
            const auto &fnodes = faces.node_lists[ num_face ];
            int nb_fnodes = faces.nb_nodes[ num_face ];
            // case handled by generated codes (makes a return if handled)
            if ( nb_fnodes <= 8 ) {
                SimdVec<std::uint64_t,8> blo = SimdVec<std::uint64_t,8>::load_aligned( fnodes.data() );
                SimdVec<std::uint64_t,8> bsh = SimdVec<std::uint64_t,8>( 1 ) << blo;
                SimdVec<std::uint64_t,8> ban = SimdVec<std::uint64_t,8>( outside_nodes ) & bsh;
                int msk = 1 << nb_fnodes, ouf = ( ban.nz() & ( msk - 1 ) ) | msk;
                #include "Internal/(ConvexPolyhedron3Lt64_plane_cut_switch.cpp).h"
            }

            TODO;
        };
        for( int n = 0; n < faces_size; ++n )
            if ( outside_nodes & faces.node_masks[ n ] )
                handle_intersected_face( n );
        //        // outside faces, SIMD version
        //        std::uint64_t ouf = 0;
        //        using NodeMask = typename Lt64FaceBlock::NodeMask;
        //        SimdRange<SimdSize<NodeMask>::value>::for_each( faces_size, [&]( int n, auto s ) {
        //            using LF = SimdVec<NodeMask,s.val>;
        //            LF nm = LF::load_aligned( faces.node_masks + n );
        //            LF an = LF( outside_nodes ) & nm;
        //            ouf |= an.nz() << n;
        //        } );
        //        for_each_nz_bit( ouf, [&]( int n ) {
        //            handle_intersected_face( n );
        //        } );

        // remove the void faces
        auto remove_void_faces = [&]() {
            for( int num_in_faces_to_rem = 0; num_in_faces_to_rem < nb_faces_to_rem; ++num_in_faces_to_rem ) {
                int num_face_to_rem = faces_to_rem[ num_in_faces_to_rem ];
                while ( true ) {
                    if ( --faces_size <= num_face_to_rem )
                        return;
                    if ( faces.node_masks[ faces_size ] )
                        break;
                }
                faces.cpy( num_face_to_rem, faces_size );
            }
        };
        remove_void_faces();

        // if non void
        if ( faces_size ) {
            // make the new face
            int nf = faces_size++;
            int nb_nodes = 0;
            std::uint64_t node_mask = 0;
            for( std::uint8_t i = last_cut_node, c = 0; ; c++ ) {
                if ( nb_nodes >= 16 )
                    additional_nums.push_back( i );
                else
                    faces.node_lists[ nf ][ nb_nodes ] = i;
                ++nb_nodes;

                node_mask |= std::uint64_t( 1 ) << i;

                std::uint8_t n = prev_cut_nodes[ i ];
                if ( n == last_cut_node )
                    break;
                i = n;
            }

            faces.node_masks[ nf ] = node_mask;
            faces.normal_xs [ nf ] = cut_dir[ 0 ][ num_cut ];
            faces.normal_ys [ nf ] = cut_dir[ 1 ][ num_cut ];
            faces.normal_zs [ nf ] = cut_dir[ 2 ][ num_cut ];
            faces.nb_nodes  [ nf ] = nb_nodes;
            faces.cut_ids   [ nf ] = cut_ids[ num_cut ];

            // repl node data
            for( int i = 0; i < nb_repl_nodes; ++i )
                nodes.local_at( repl_node_dsts[ i ] ).get_straight_content_from( nodes.local_at( repl_node_srcs[ i ] ) );

            // if we have nodes to free
            if ( available_nodes ) {
                // while we have nodes to free
                std::uint64_t moved_nodes = 0;
                do {
                    // if the last node is outside, remove it from cou. Else, move if to the free room
                    std::uint64_t moved_node = std::uint64_t( 1 ) << --nodes_size;
                    if ( outside_nodes & moved_node ) {
                        available_nodes -= moved_node;
                    } else {
                        int n = tzcnt( available_nodes );
                        moved_nodes |= moved_node;
                        repl_node_dsts[ nodes_size ] = n;
                        available_nodes -= std::uint64_t( 1 ) << n;
                        nodes.local_at( n ).get_straight_content_from( nodes.local_at( nodes_size ) );
                    }
                } while ( available_nodes );

                // move the nodes
                auto move_nodes_in_face = [&]( unsigned num_face ) {
                    auto fm = faces.node_masks[ num_face ];

                    for( int i = 0; i < faces.nb_nodes[ num_face ]; ++i ) {
                        auto &nn = faces.node_lists[ num_face ][ i ];
                        std::uint64_t msk = std::uint64_t( 1 ) << nn;
                        if ( moved_nodes & msk ) {
                            fm -= msk;
                            nn = repl_node_dsts[ nn ];
                            fm |= std::uint64_t( 1 ) << nn;
                        }
                    }

                    faces.node_masks[ num_face ] = fm;
                };
                // TODO: optimize with SIMD
                for( int num_face = 0; num_face < faces_size; ++num_face )
                    if ( moved_nodes & faces.node_masks[ num_face ] )
                        move_nodes_in_face( num_face );
            }
        } else {
            nodes_size = 0;
        }

        // if it does not fit
        if ( additional_nums.size() || additional_nodes.size() )
            return num_cut;
    }

    return nb_cuts;
}

template<class Pc>
std::size_t ConvexPolyhedron3<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ) {
    return plane_cut( cut_dir, cut_ps, cut_id, nb_cuts, N<0>() );
}

template<class Pc>
typename ConvexPolyhedron3<Pc>::TF ConvexPolyhedron3<Pc>::integral() const {
    TODO;
    return 0;
}

} // namespace sdot
