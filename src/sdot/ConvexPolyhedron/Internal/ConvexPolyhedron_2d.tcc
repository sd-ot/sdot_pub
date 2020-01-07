#include "../../Support/aligned_memory.h"
#include "../../Support/bit_handling.h"
#include "../../Support/SimdRange.h"
#include "../../Support/SimdVec.h"
#include "../../Support/Simplex.h"
#include "../../Support/ASSERT.h"
#include "../../Support/TODO.h"
#include "../../Support/P.h"
#include "ConvexPolyhedron_2d.h"

#include "ConvexPolyhedron2dLt64_cut_gen_gen.h"

namespace sdot {

template<class Pc>
ConvexPolyhedron<Pc,2,void>::ConvexPolyhedron( Pt pmin, Pt pmax, CI cut_id ) : ConvexPolyhedron() {
    // size (assuming that rese is enough)
    nodes_size = 4;

    // points
    pxs[ 0 ] = pmin[ 0 ]; pys[ 0 ] = pmin[ 1 ];
    pxs[ 1 ] = pmax[ 0 ]; pys[ 1 ] = pmin[ 1 ];
    pxs[ 2 ] = pmax[ 0 ]; pys[ 2 ] = pmax[ 1 ];
    pxs[ 3 ] = pmin[ 0 ]; pys[ 3 ] = pmax[ 1 ];

    // normals
    nxs[ 0 ] =  0; nys[ 0 ] = -1;
    nxs[ 1 ] = +1; nys[ 1 ] =  0;
    nxs[ 2 ] =  0; nys[ 2 ] = +1;
    nxs[ 3 ] = -1; nys[ 3 ] =  0;

    // cut_ids
    cis[ 0 ] = cut_id;
    cis[ 1 ] = cut_id;
    cis[ 2 ] = cut_id;
    cis[ 3 ] = cut_id;
}

template<class Pc>
ConvexPolyhedron<Pc,2,void>::ConvexPolyhedron() {
    nodes_rese = 128;
    nodes_size = 0;

    pxs = new ( aligned_malloc( nodes_rese * sizeof( TF ), 64 ) ) TF[ nodes_rese ];
    pys = new ( aligned_malloc( nodes_rese * sizeof( TF ), 64 ) ) TF[ nodes_rese ];
    nxs = new ( aligned_malloc( nodes_rese * sizeof( TF ), 64 ) ) TF[ nodes_rese ];
    nys = new ( aligned_malloc( nodes_rese * sizeof( TF ), 64 ) ) TF[ nodes_rese ];
    cis = new ( aligned_malloc( nodes_rese * sizeof( CI ), 64 ) ) TF[ nodes_rese ];
    ds  = new ( aligned_malloc( nodes_rese * sizeof( TF ), 64 ) ) TF[ nodes_rese ];
}

template<class Pc>
ConvexPolyhedron<Pc,2,void>::~ConvexPolyhedron() {
    for( std::size_t i = 0; i < nodes_size; ++i ) {
        pxs[ i ].~TF();
        pys[ i ].~TF();
        nxs[ i ].~TF();
        nys[ i ].~TF();
        cis[ i ].~CI();
        ds [ i ].~TF();
    }

    aligned_free( pxs );
    aligned_free( pys );
    aligned_free( nxs );
    aligned_free( nys );
    aligned_free( cis );
    aligned_free( ds  );
}

template<class Pc>
ConvexPolyhedron<Pc,2,void> &ConvexPolyhedron<Pc,2,void>::operator=( const ConvexPolyhedron &that ) {
    reserve_without_cp( that.nodes_size );
    nodes_size = that.nodes_size;

    // vetorized by the compiler
    for( std::size_t i = 0; i < nodes_size; ++i ) {
        pxs[ i ] = that.pxs[ i ];
        pys[ i ] = that.pys[ i ];
        nxs[ i ] = that.nxs[ i ];
        nys[ i ] = that.nys[ i ];
        cis[ i ] = that.cis[ i ];
        ds [ i ] = that.ds [ i ];
    }

    return *this;
}

template<class Pc>
void ConvexPolyhedron<Pc,2,void>::write_to_stream( std::ostream &os ) const {
    os << "pos: ";
    for( int i = 0; i < nb_nodes(); ++i )
        os << ( i ? " [" : "[" ) << pxs[ i ] << " " << pys[ i ] << "]";
}

template<class Pc> template<class F>
void ConvexPolyhedron<Pc,2,void>::for_each_bound( const F &/*f*/ ) const {
    //    for( int n0 = nodes_size - 1, n1 = 0; n1 < nodes_size; n0 = n1++ )
    //        f( Bound{ .nodes = &nodes, .n0 = n0, .n1 = n1 } );
    TODO;
}

template<class Pc> template<class F>
void ConvexPolyhedron<Pc,2,void>::for_each_node( const F &f ) const {
    for( int i = 0; i < nb_nodes(); ++i )
        f( node( i ) );
}

template<class Pc>
int ConvexPolyhedron<Pc,2,void>::nb_nodes() const {
    return nodes_size;
}

template<class Pc>
bool ConvexPolyhedron<Pc,2,void>::empty() const {
    return nodes_size == 0;
}

template<class Pc>
typename ConvexPolyhedron<Pc,2,void>::Pt ConvexPolyhedron<Pc,2,void>::node( int index ) const {
    return { pxs[ index ], pys[ index ] };
}

template<class Pc> template<int flags>
void ConvexPolyhedron<Pc,2,void>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_ids, int nb_cuts, N<flags> ) {
    // max 8 nodes version (we store coords in registers)
    int num_cut = 0;
    if ( nodes_size <= 8 ) {
        if ( internal::ConvexPolyhedron2dLt64_cut( num_cut, pxs, pys, reinterpret_cast<std::size_t *>( cis ), nodes_size, cut_dir[ 0 ], cut_dir[ 1 ], cut_ps, reinterpret_cast<const std::size_t *>( cut_ids ), nb_cuts ) )
            return;
    }

    TODO;
    //    // => more than 8 nodes, but less than 65
    //    for( ; ; ++num_cut ) {
    //        if ( num_cut == nb_cuts )
    //            return fu( *this );

    //        TF cx = cut_dir[ 0 ][ num_cut ];
    //        TF cy = cut_dir[ 1 ][ num_cut ];
    //        TF cs = cut_ps[ num_cut ];


    //        // get distance and outside bit for each node
    //        std::uint64_t outside_nodes = 0;
    //        alignas( 64 ) TF distances[ 64 ];
    //        constexpr int ss = SimdSize<TF>::value;
    //        SimdRange<ss>::for_each( nodes_size, [&]( int n, auto s ) {
    //            using LF = SimdVec<TF,s.val>;

    //            LF px = LF::load_aligned( nodes.xs + n );
    //            LF py = LF::load_aligned( nodes.ys + n );
    //            LF bi = px * LF( cx ) + py * LF( cy );

    //            LF::store_aligned( distances + n, bi - LF( cs ) );
    //            std::uint64_t lo = bi > LF( cs );
    //            outside_nodes |= lo << n;
    //        } );

    //        // if nothing has changed => go to the next cut
    //        if ( outside_nodes == 0 )
    //            continue;

    //        // make a new edge set, in tmp storage
    //        std::bitset<64> outside_vec = outside_nodes;
    //        int new_nodes_size = 0;
    //        CI new_cut_ids[ 128 ];
    //        TF new_xs[ 128 ];
    //        TF new_ys[ 128 ];

    //        for( int n0 = 0, nm = nodes_size - 1; n0 < nodes_size; nm = n0++ ) {
    //            if ( outside_vec[ n0 ] )
    //                continue;

    //            if ( outside_vec[ nm ] ) {
    //                TF m = distances[ n0 ] / ( distances[ nm ] - distances[ n0 ] );
    //                new_xs[ new_nodes_size ] = nodes.xs[ n0 ] - m * ( nodes.xs[ nm ] - nodes.xs[ n0 ] );
    //                new_ys[ new_nodes_size ] = nodes.ys[ n0 ] - m * ( nodes.ys[ nm ] - nodes.ys[ n0 ] );
    //                new_cut_ids[ new_nodes_size ] = nodes.cut_ids[ nm ];
    //                ++new_nodes_size;
    //            }

    //            new_cut_ids[ new_nodes_size ] = nodes.cut_ids[ n0 ];
    //            new_xs[ new_nodes_size ] = nodes.xs[ n0 ];
    //            new_ys[ new_nodes_size ] = nodes.ys[ n0 ];
    //            ++new_nodes_size;

    //            int n1 = ( n0 + 1 ) % nodes_size;
    //            if ( outside_vec[ n1 ] ) {
    //                TF m = distances[ n0 ] / ( distances[ n1 ] - distances[ n0 ] );
    //                new_xs[ new_nodes_size ] = nodes.xs[ n0 ] - m * ( nodes.xs[ n1 ] - nodes.xs[ n0 ] );
    //                new_ys[ new_nodes_size ] = nodes.ys[ n0 ] - m * ( nodes.ys[ n1 ] - nodes.ys[ n0 ] );
    //                new_cut_ids[ new_nodes_size ] = cut_ids[ num_cut ];
    //                ++new_nodes_size;
    //            }
    //        }

    //        if ( new_nodes_size > 64 ) {
    //            // update cp_gen
    //            cp_gen.nodes.resize( new_nodes_size );
    //            for( int i = 0; i < new_nodes_size; ++i )
    //                cp_gen.nodes[ i ] = { .p = { new_xs[ i ], new_ys[ i ] }, .cut_id = new_cut_ids[ i ] };

    //            // make the plane_cut on cp_gen
    //            ++num_cut;
    //            for( int d = 0; d < dim; ++d )
    //                cut_dir[ d ] += num_cut;
    //            cut_ps  += num_cut;
    //            cut_ids += num_cut;
    //            nb_cuts -= num_cut;
    //            return cp_gen.plane_cut( cut_dir, cut_ps, cut_ids, nb_cuts, N<flags>(), fu );
    //        }

    //        std::memcpy( nodes.cut_ids, new_cut_ids, sizeof( CI ) * new_nodes_size );
    //        std::memcpy( nodes.xs, new_xs, sizeof( TF ) * new_nodes_size );
    //        std::memcpy( nodes.ys, new_ys, sizeof( TF ) * new_nodes_size );
    //        nodes_size = new_nodes_size;
    //    }
}

template<class Pc>
void ConvexPolyhedron<Pc,2,void>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, int nb_cuts ) {
    return plane_cut( cut_dir, cut_ps, cut_id, nb_cuts, N<0>() );
}

//template<class Pc> template<class TL>
//void ConvexPolyhedron<Pc,2,void>::Bound::for_each_simplex( const TL &f ) const {
//    f( Simplex<TF,2,1>{ Pt{ nodes->xs[ n0 ], nodes->ys[ n0 ] }, Pt{ nodes->xs[ n1 ], nodes->ys[ n1 ] } } );
//}

//template<class Pc>
//typename Pc::CI ConvexPolyhedron<Pc,2,void>::Bound::cut_id() const {
//    return nodes->cut_ids[ n0 ];
//}

} // namespace sdot
