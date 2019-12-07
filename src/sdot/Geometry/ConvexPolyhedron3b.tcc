//#include "../Support/Display/generic_ostream_output.h"
#include "../Support/Display/binary_repr.h"
#include "../Support/aligned_memory.h"
#include "../Support/bit_handling.h"
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
    auto set_node = [&]( TI index, TF x, TF y, TF z ) -> Lt64NodeBlock * {
        Lt64NodeBlock &res = nodes.local_at( index );
        res.x = x;
        res.y = y;
        res.z = z;

        return &res;
    };

    Lt64NodeBlock *n[ 8 ] = {
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
    nodes_size    = 0;
    faces_size    = 0;
    num_cut       = 0;

    for( std::size_t i = 0; i < max_nb_edges; ++i )
        edge_num_cuts[ i ] = 0;
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
        face.foreach_node_index( [&]( TI index ) {
            os << " " << index;
        } );
    } );
}

template<class Pc>
void ConvexPolyhedron3<Pc>::display_vtk( VtkOutput &vo, const std::vector<TF> &cell_values, Pt offset, bool /*display_both_sides*/ ) const {
    std::vector<VtkOutput::Pt> pts;
    for_each_face( [&]( const Face &face ) {
        if ( allow_ball_cut ) { //  && face.round
            TODO;
        } else /*if ( display_both_sides || face.cut_id > sphere_cut_id )*/ {
            pts.clear();
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
void ConvexPolyhedron3<Pc>::for_each_face( const std::function<void( const Face &)> &f ) const {
    for( unsigned num_face = 0; num_face < faces_size; ++num_face )
        f( { num_face, this } );
}


template<class Pc> template<int flags>
void ConvexPolyhedron3<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ) {
    constexpr int s = SimdSize<TF>::value;
    using SF = SimdVec<TF,s>;

    for( std::size_t num_cut = 0; num_cut < nb_cuts; ++num_cut ) {
        // get distances and bits for outside nodes
        TF cx = cut_dir[ 0 ][ num_cut ];
        TF cy = cut_dir[ 1 ][ num_cut ];
        TF cz = cut_dir[ 2 ][ num_cut ];
        TF cs = cut_ps[ num_cut ];
        SF nx = cx;
        SF ny = cy;
        SF nz = cz;
        SF ns = cs;
        std::uint64_t ou = 0;
        unsigned n = 0;
        for( ; n + 8 <= nodes_size; n += 8 ) {
            SF px = SF::load_aligned( &nodes.x + n );
            SF py = SF::load_aligned( &nodes.y + n );
            SF pz = SF::load_aligned( &nodes.z + n );
            SF bi = px * nx + py * ny + pz * nz;
            std::uint64_t lo = bi > ns;
            SF::store_aligend( &nodes.d, bi - ns );
            ou |= lo << n;
        }
        for( ; n < nodes_size; ++n ) {
            TF px = (&nodes.x)[ n ];
            TF py = (&nodes.y)[ n ];
            TF pz = (&nodes.z)[ n ];
            TF bi = px * cx + py * cy + pz * cz;
            std::uint64_t lo = bi > cs;
            (&nodes.d)[ n ] = bi - cs;
            ou |= lo << n;
        }
        
        // if nothing has changed, we can go to the next cut
        if ( ou == 0 )
            continue;

        // else, find the intersected faces
        unsigned faces_to_rem[ Lt64FaceBlock::max_nb_faces_per_cell ];
        unsigned nb_faces_to_rem = 0;
        ++num_cut;

        // handke intersected faces
        auto handle_intersected_face = [&]( unsigned num_face ) {
            // make a cut id: 4 bits for the initial number of nodes, 16 bits for outsideness of each node
            const auto &node_list = faces.node_lists[ n ];
            unsigned nb_nodes = faces.nb_nodes[ num_face ];
            if ( nb_nodes > 8 )
                TODO;

            // unsigned ouf = 0;
            // for( std::size_t num_node = 0; num_node < nb_nodes; ++num_node ) {
            //     bool out = ( 1 << node_list[ num_node ] ) & ou;
            //     ouf |= unsigned( out ) << num_node;
            // }
            __m128i blo = _mm_load_si128( reinterpret_cast<const __m128i *>( node_list.data() ) ); // load the 16 indices (in 8 bit)
            __m512i bou = _mm512_set1_epi64( ou );
            __m512i bex = _mm512_cvtepi8_epi64( blo ); // 8 bits to 64 bits indices (first part)
            __m512i bsh = _mm512_sllv_epi64( _mm512_set1_epi64( 1 ), bex ); // load the first 8 indices to 64 bits
            __m512i ban = _mm512_and_epi64( bsh, bou );
            std::uint16_t ouf = ( _mm512_cmpneq_epi64_mask( ban, _mm512_setzero_si512() ) << 3 ) + ( nb_nodes - 3 );
            do {
                #include "Internal/(ConvexPolyhedron3Lt64_plane_cut_switch.cpp).h"
            } while ( 0 );
        };

        n = 0;
        // __m512i mn = _mm512_set1_epi64( ou );
        //        for( ; n + 8 <= faces_size; n += 8 ) {
        //            //             __m512i ms = _mm512_load_epi64( faces.node_masks + n );
        //            //             __m512i am = _mm512_and_epi64( mn, ms );
        //            //             std::uint8_t lo = _mm512_cmp_pd_mask( bi, rd, _CMP_GT_OQ );
        //            TODO;
        //        }
        for( ; n < faces_size; ++n )
            if ( ou & faces.node_masks[ n ] )
                handle_intersected_face( n );
        
        // remove void faces
        auto remove_void_faces = [&]() {
            for( unsigned num_in_faces_to_rem = 0; num_in_faces_to_rem < nb_faces_to_rem; ++num_in_faces_to_rem ) {
                unsigned num_face_to_rem = faces_to_rem[ num_in_faces_to_rem ];
                while ( true ) {
                    if ( --faces_size <= num_face_to_rem )
                        return;
                    if ( faces.node_masks[ faces_size ] )
                        break;
                }

                faces.node_masks[ num_face_to_rem ] = faces.node_masks[ faces_size ];
                faces.node_lists[ num_face_to_rem ] = faces.node_lists[ faces_size ];
                faces.normal_xs [ num_face_to_rem ] = faces.normal_xs [ faces_size ];
                faces.normal_ys [ num_face_to_rem ] = faces.normal_ys [ faces_size ];
                faces.normal_zs [ num_face_to_rem ] = faces.normal_zs [ faces_size ];
                faces.nb_nodes  [ num_face_to_rem ] = faces.nb_nodes  [ faces_size ];
                faces.cut_ids   [ num_face_to_rem ] = faces.cut_ids   [ faces_size ];
            }
        };
        remove_void_faces();
    }
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

} // namespace sdot
