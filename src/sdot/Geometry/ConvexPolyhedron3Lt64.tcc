#include "../Support/Display/binary_repr.h"
#include "../Support/aligned_memory.h"
#include "../Support/bit_handling.h"
#include "../Support/SimdRange.h"
#include "../Support/SimdVec.h"
#include "../Support/ASSERT.h"
#include "../Support/TODO.h"
#include "../Support/pi.h"
#include "../Support/P.h"
#include "ConvexPolyhedron3Lt64.h"
#include <cstring>
#include <iomanip>
#include <bitset>
#include <map>
#include <set>

namespace sdot {

template<class Pc>
ConvexPolyhedron3Lt64<Pc>::Box::Box( Pt p0, Pt p1, CI cut_id ) : cp( p0, p1, cut_id ) {
}

template<class Pc>
ConvexPolyhedron3Lt64<Pc>::ConvexPolyhedron3Lt64( Pt p0, Pt p1, CI cut_id ) : ConvexPolyhedron3Lt64() {
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
ConvexPolyhedron3Lt64<Pc>::ConvexPolyhedron3Lt64() {
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
ConvexPolyhedron3Lt64<Pc>::~ConvexPolyhedron3Lt64() {
}

template<class Pc>
ConvexPolyhedron3Lt64<Pc> &ConvexPolyhedron3Lt64<Pc>::operator=( const ConvexPolyhedron3Lt64 &that ) {
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
ConvexPolyhedron3Lt64<Pc> &ConvexPolyhedron3Lt64<Pc>::operator=( const Box &that ) {
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
void ConvexPolyhedron3Lt64<Pc>::write_to_stream( std::ostream &os, bool /*debug*/ ) const {
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
void ConvexPolyhedron3Lt64<Pc>::display_vtk( VtkOutput &vo, const std::vector<TF> &cell_values, Pt offset, bool /*display_both_sides*/ ) const {
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
typename ConvexPolyhedron3Lt64<Pc>::TI ConvexPolyhedron3Lt64<Pc>::nb_nodes() const {
    return nodes_size;
}

template<class Pc>
bool ConvexPolyhedron3Lt64<Pc>::empty() const {
    return nodes_size == 0;
}

template<class Pc>
const typename ConvexPolyhedron3Lt64<Pc>::Node &ConvexPolyhedron3Lt64<Pc>::node( TI index ) const {
    return nodes.local_at( index );
}

template<class Pc>
typename ConvexPolyhedron3Lt64<Pc>::Node &ConvexPolyhedron3Lt64<Pc>::node( TI index ) {
    return nodes.local_at( index );
}

template<class Pc>
void ConvexPolyhedron3Lt64<Pc>::for_each_boundary_item( const std::function<void( const BoundaryItem &boundary_item )> &f ) const {
    for_each_face( f );
}

template<class Pc>
void ConvexPolyhedron3Lt64<Pc>::for_each_face( const std::function<void( const Face & )> &f ) const {
    for( int num_face = 0; num_face < faces_size; ++num_face )
        f( { num_face, this } );
}

template<class Pc>
void ConvexPolyhedron3Lt64<Pc>::for_each_node( const std::function<void( const Node & )> &f ) const {
    for( int num_node = 0; num_node < nodes_size; ++num_node )
        f( nodes.local_at( num_node ) );
}


template<class Pc> template<int flags>
std::size_t ConvexPolyhedron3Lt64<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_ids, std::size_t nb_cuts, N<flags> ) {
    constexpr int ss = SimdSize<TF>::value;
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
        int ind_nxt_tmp_node = 3 * max_nb_nodes;

        // linked list of nodes for the new face
        std::uint8_t prev_cut_nodes[ max_nb_nodes ]; // index to the next node
        //std::uint8_t last_cut_node;

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

            // generic case (several cuts, nb_nodes > 8, etc...)
            std::array<std::uint8_t,2*Lt64FaceBlock::max_nb_nodes_per_face> new_fnode_lists;
            std::array<std::uint8_t,3*Lt64FaceBlock::max_nb_nodes_per_cell> prev_cut_node;
            std::uint64_t new_fnode_mask = 0;
            int new_nb_fnodes = 0;

            // helper
            auto add_node_between = [&]( int num_node_0, int num_node_1 ) {
                int min_node_128 = std::min( num_node_0, num_node_1 );
                int max_node_128 = std::max( num_node_0, num_node_1 );
                int num_edge_128 = 64 * max_node_128 + min_node_128;

                int num_node_128;
                if ( edge_num_cut_procs[ num_edge_128 ] != num_cut_proc ) {
                    int pos_node;
                    if ( available_nodes ) { // there's a node that is going to be freed
                        int nn = tzcnt( available_nodes );
                        available_nodes -= std::uint64_t( 1 ) << nn;

                        pos_node = ind_nxt_tmp_node++;
                        num_node_128 = nn;

                        repl_node_dsts[ nb_repl_nodes ] = num_node_128;
                        repl_node_srcs[ nb_repl_nodes ] = pos_node;
                        ++nb_repl_nodes;
                    } else {
                        num_node_128 = nodes_size;
                        pos_node = nodes_size++;
                    }
                    const Node &n0 = nodes.local_at( num_node_0 );
                    const Node &n1 = nodes.local_at( num_node_1 );
                    nodes.local_at( pos_node ).set_pos( n0.pos() + n0.d / ( n0.d - n1.d ) * ( n1.pos() - n0.pos() ) );
                    edge_num_cut_procs[ num_edge_128 ] = num_cut_proc;
                    edge_cuts[ num_edge_128 ] = num_node_128;
                } else
                    num_node_128 = edge_cuts[ num_edge_128 ];

                new_fnode_lists[ new_nb_fnodes++ ] = num_node_128;
                new_fnode_mask |= std::uint64_t( 1 ) << num_node_128;
                return num_node_128;
            };


            // look up for a node after an outside one
            int o0 = 0;
            while ( ( outside_nodes & ( std::uint64_t( 1 ) << fnodes[ o0 ] ) ) == 0 )
                if ( ++o0 == nb_fnodes )
                    break;
            ++o0;

            // remake the face
            int prev_out_in;
            for( int i0 = o0; i0 < o0 + nb_fnodes; ++i0 ) {
                int n0 = fnodes[ ( i0 + 0 ) % nb_fnodes ];
                std::uint64_t m0 = std::uint64_t( 1 ) << n0;
                if ( outside_nodes & m0 )
                    continue;

                // outside (nm) => inside (n0)
                int nm = fnodes[ ( i0 - 1 ) % nb_fnodes ];
                std::uint64_t mm = std::uint64_t( 1 ) << nm;
                if ( outside_nodes & mm )
                    prev_out_in = add_node_between( nm, n0 );

                //
                new_fnode_lists[ new_nb_fnodes++ ] = n0;
                new_fnode_mask |= m0;

                // inside (n0) => outside (n1)
                int n1 = fnodes[ ( i0 + 1 ) % nb_fnodes ];
                std::uint64_t m1 = std::uint64_t( 1 ) << n1;
                if ( outside_nodes & m1 )
                    prev_cut_node[ prev_out_in ] = add_node_between( n0, n1 );
            }

            if ( nb_out_ins ) {
                if ( nb_out_ins > 1 )
                    TODO;
                prev_cut_nodes[ out_ins[ 0 ] ] = in_outs[ 0 ];
                last_cut_node = out_ins[ 0 ];
            }

            // save the face
            if ( new_nb_fnodes <= Lt64FaceBlock::max_nb_nodes_per_face ) {
                for( int i = 0; i < new_nb_fnodes; ++i )
                    faces.node_lists[ num_face ][ i ] = new_fnode_lists[ i ];
            } else {
                for( int i = 0; i < Lt64FaceBlock::max_nb_nodes_per_face; ++i )
                    faces.node_lists[ num_face ][ i ] = new_fnode_lists[ i ];

                faces.off_in_ans[ num_face ] = additional_nums.size();
                for( int i = Lt64FaceBlock::max_nb_nodes_per_face; i < new_nb_fnodes; ++i )
                    additional_nums.push_back( new_fnode_lists[ i ] );
            }
            faces.node_masks[ num_face ] = new_fnode_mask;
            faces.nb_nodes[ num_face ] = new_nb_fnodes;
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

        // if cell is not void
        if ( faces_size ) {
            // make the new face
            int nf = faces_size++;
            int nb_nodes = 0;
            std::uint64_t node_mask = 0;
            for( std::uint8_t i = last_cut_node, c = 0; ; c++ ) {
                if ( nb_nodes >= 16 ) {
                    if ( nb_nodes == 16 )
                        faces.off_in_ans[ nf ] = additional_nums.size();
                    additional_nums.push_back( i );
                } else
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

            // if it does not fit
            if ( nodes_size > 64 || additional_nums.size() )
                return num_cut;
        } else {
            nodes_size = 0;
        }
    }

    return nb_cuts;
}

template<class Pc>
std::size_t ConvexPolyhedron3Lt64<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ) {
    return plane_cut( cut_dir, cut_ps, cut_id, nb_cuts, N<0>() );
}

template<class Pc>
bool ConvexPolyhedron3Lt64<Pc>::valid() const {
    for( int num_face = 0; num_face < faces_size; ++num_face ) {
        using M = std::uint64_t;
        M msk = 0;
        for( int nif = 0; nif < faces.nb_nodes[ num_face ]; ++nif )
            msk |= M( 1 ) << faces.node_lists[ num_face ][ nif ];
        if ( msk != faces.node_masks[ num_face ] )
            ERROR( "..." );
    }
    return true;
}

} // namespace sdot
