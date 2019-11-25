//#include "../Support/Display/generic_ostream_output.h"
#include "../Support/Display/binary_repr.h"
#include "../Support/aligned_memory.h"
#include "../Support/bit_handling.h"
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
ConvexPolyhedron3<Pc>::ConvexPolyhedron3( const Box &box ) : ConvexPolyhedron3() {
    set_box( box );
}

template<class Pc>
ConvexPolyhedron3<Pc>::ConvexPolyhedron3() {
    sphere_radius = 0;
    nodes_size    = 0;
    nodes_rese    = 0;
    faces_size    = 0;
    faces_rese    = 0;
}


template<class Pc>
void ConvexPolyhedron3<Pc>::set_box( const Box &box ) {
    static double junx[] = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 };
    static double juny[] = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 };
    static double junz[] = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 };
    static double juna[] = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 };
    static double junb[] = { 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4 };

    memcpy( __builtin_assume_aligned( &lt64_node_block.x, 64 ), __builtin_assume_aligned( junx, 64 ), 8 * sizeof( double ) );
    memcpy( __builtin_assume_aligned( &lt64_node_block.y, 64 ), __builtin_assume_aligned( juny, 64 ), 8 * sizeof( double ) );
    memcpy( __builtin_assume_aligned( &lt64_node_block.z, 64 ), __builtin_assume_aligned( junz, 64 ), 8 * sizeof( double ) );

    memcpy( __builtin_assume_aligned( &lt64_face_block.node_mask, 64 ), __builtin_assume_aligned( juna, 64 ), 6 * sizeof( double ) );
    memcpy( __builtin_assume_aligned( &lt64_face_block.node_lst0, 64 ), __builtin_assume_aligned( junb, 64 ), 6 * sizeof( double ) );

    memcpy( __builtin_assume_aligned( &nodes_size, 64 ), __builtin_assume_aligned( junb, 64 ), 2 * sizeof( double ) );

    //    // nodes
    //    nodes_size = 8;
    //    faces_size = 6;

    //    auto set_node = [&]( TI index, TF x, TF y, TF z ) -> Lt64NodeBlock * {
    //        Lt64NodeBlock &res = lt64_node_block.local_at( index );
    //        res.x = x;
    //        res.y = y;
    //        res.z = z;

    //        return &res;
    //    };

    //    Lt64NodeBlock *n[ 8 ] = {
    //        set_node( 0, box.p0.x, box.p0.y, box.p0.z ),
    //        set_node( 1, box.p1.x, box.p0.y, box.p0.z ),
    //        set_node( 2, box.p0.x, box.p1.y, box.p0.z ),
    //        set_node( 3, box.p1.x, box.p1.y, box.p0.z ),
    //        set_node( 4, box.p0.x, box.p0.y, box.p1.z ),
    //        set_node( 5, box.p1.x, box.p0.y, box.p1.z ),
    //        set_node( 6, box.p0.x, box.p1.y, box.p1.z ),
    //        set_node( 7, box.p1.x, box.p1.y, box.p1.z )
    //    };

    //    // faces
    //    auto add_face = [&]( TI index, int n0, int n1, int n2, int n3, Pt normal ) {
    //        constexpr std::uint64_t m = 1;

    //        Lt64FaceBlock &res = lt64_face_block.local_at( index );
    ////        res.normal_x = normal.x;
    ////        res.normal_y = normal.y;
    ////        res.normal_z = normal.z;

    //        res.node_lst0[ 0 ] = n0;
    //        res.node_lst0[ 1 ] = n1;
    //        res.node_lst0[ 2 ] = n2;
    //        res.node_lst0[ 3 ] = n3;
    //        res.node_lst0[ 4 ] = 255u;

    //        res.node_mask =
    //                ( m << n0 ) |
    //                ( m << n1 ) |
    //                ( m << n2 ) |
    //                ( m << n3 ) ;
    //    };

    //    add_face( 0,  0, 2, 3, 1,  { 0, 0, -1 } );
    //    add_face( 1,  4, 5, 7, 6,  { 0, 0, +1 } );
    //    add_face( 2,  0, 1, 5, 4,  { 0, -1, 0 } );
    //    add_face( 3,  2, 6, 7, 3,  { 0, +1, 0 } );
    //    add_face( 4,  0, 4, 6, 2,  { -1, 0, 0 } );
    //    add_face( 5,  1, 3, 7, 5,  { +1, 0, 0 } );
}

template<class Pc>
ConvexPolyhedron3<Pc>::~ConvexPolyhedron3() {
    //    if ( nodes_rese )
    //        aligned_free( nodes );
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
    set_box( box );
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
//    std::vector<VtkOutput::Pt> pts;
//    for_each_face( [&]( const Face &face ) {
//        if ( allow_ball_cut && face.round ) {
//            TODO;
//        } else /*if ( display_both_sides || face.cut_id > sphere_cut_id )*/ {
//            pts.clear();
//            face.foreach_node( [&]( const Node &node ) {
//                pts.push_back( node.pos() + offset );
//            } );
//            vo.add_polygon( pts, cell_values );
//        }
//    } );
}


template<class Pc> template<class F>
void ConvexPolyhedron3<Pc>::for_each_face( const F &f ) const {
    for( TI i = 0; i < faces_size; ++i )
        f( lt64_face_block.local_at( i ) );
}

template<class Pc> template<class F>
void ConvexPolyhedron3<Pc>::for_each_node( const F &f ) const {
    TI s = nodes_size;
    if ( s <= 64 ) {
        for( TI i = 0; i < s; ++i )
            f( lt64_node_block.local_at( i ) );
    } else {
        TODO;
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

//template<class Pc>
//const typename ConvexPolyhedron3<Pc>::Node &ConvexPolyhedron3<Pc>::node( TI index ) const {
//    return nodes->global_at( index );
//}

//template<class Pc>
//typename ConvexPolyhedron3<Pc>::Node &ConvexPolyhedron3<Pc>::node( TI index ) {
//    return nodes->global_at( index );
//}

//template<class Pc>
//void ConvexPolyhedron3<Pc>::for_each_boundary_item( const std::function<void( const BoundaryItem &boundary_item )> &f ) const {
//    for_each_face( f );
//}

//template<class Pc>
//void ConvexPolyhedron3<Pc>::rese_nb_nodes( TI new_size ) {
//    if ( new_size > nodes_rese ) {
//        TI old_nb_blocks = nodes_rese / block_size;
//        Node *old_nodes = nodes;

//        TI nb_blocks = ( new_size + block_size - 1 ) / block_size;
//        nodes_rese = nb_blocks * block_size;

//        nodes = reinterpret_cast<Node *>( aligned_malloc( nb_blocks * sizeof( Node ), 64 ) );
//        for( TI i = 0; i < old_nb_blocks; ++i )
//            new ( nodes + i ) Node( std::move( old_nodes[ i ] ) );

//        if ( old_nb_blocks )
//            aligned_free( old_nodes );

//        TODO; // pointers
//    }
//}

//template<class Pc>
//void ConvexPolyhedron3<Pc>::set_nb_nodes( TI new_size ) {
//    rese_nb_nodes( new_size );
//    nodes_size = new_size;
//}

//template<class Pc>
//typename ConvexPolyhedron3<Pc>::Node *ConvexPolyhedron3<Pc>::new_node( Pt pos ) {
//    TI n = nb_nodes();
//    set_nb_nodes( n + 1 );

//    Node *res = &node( n );
//    res->repl.set( res );
//    res->set_pos( pos );
//    return res;
//}


template<class Pc> template<int flags>
void ConvexPolyhedron3<Pc>::plane_cut_mt_64( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ) {
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
