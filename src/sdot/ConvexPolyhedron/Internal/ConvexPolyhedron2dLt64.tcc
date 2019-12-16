#include "../../Support/Display/binary_repr.h"
#include "../../Support/bit_handling.h"
#include "../../Support/SimdRange.h"
#include "../../Support/SimdVec.h"
#include "../../Support/Simplex.h"
#include "../../Support/ASSERT.h"
#include "../../Support/TODO.h"
#include "../../Support/P.h"
#include "ConvexPolyhedron2dLt64.h"
#include <cstring>

namespace sdot {

template<class Pc>
ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::ConvexPolyhedron( Pt pmin, Pt pmax, CI cut_id ) : ConvexPolyhedron() {
    // size
    nodes_size = 4;

    // points
    nodes.xs[ 0 ] = pmin[ 0 ]; nodes.ys[ 0 ] = pmin[ 1 ];
    nodes.xs[ 1 ] = pmax[ 0 ]; nodes.ys[ 1 ] = pmin[ 1 ];
    nodes.xs[ 2 ] = pmax[ 0 ]; nodes.ys[ 2 ] = pmax[ 1 ];
    nodes.xs[ 3 ] = pmin[ 0 ]; nodes.ys[ 3 ] = pmax[ 1 ];

    // normals
    nodes.dir_xs[ 0 ] =  0; nodes.dir_ys[ 0 ] = -1;
    nodes.dir_xs[ 1 ] = +1; nodes.dir_ys[ 1 ] =  0;
    nodes.dir_xs[ 2 ] =  0; nodes.dir_ys[ 2 ] = +1;
    nodes.dir_xs[ 3 ] = -1; nodes.dir_ys[ 3 ] =  0;

    // cut_ids
    nodes.cut_ids[ 0 ] = cut_id;
    nodes.cut_ids[ 1 ] = cut_id;
    nodes.cut_ids[ 2 ] = cut_id;
    nodes.cut_ids[ 3 ] = cut_id;
}

template<class Pc>
ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::ConvexPolyhedron() {
    nodes_size = 0;
}

template<class Pc>
ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64> &ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::operator=( const ConvexPolyhedron &that ) {
    nodes_size = that.nodes_size;

    std::memcpy( nodes.xs     , that.nodes.xs     , that.nodes_size * sizeof( TF ) );
    std::memcpy( nodes.ys     , that.nodes.ys     , that.nodes_size * sizeof( TF ) );
    std::memcpy( nodes.dir_ys , that.nodes.dir_xs , that.nodes_size * sizeof( TF ) );
    std::memcpy( nodes.dir_ys , that.nodes.dir_ys , that.nodes_size * sizeof( TF ) );
    std::memcpy( nodes.cut_ids, that.nodes.cut_ids, that.nodes_size * sizeof( CI ) );

    return *this;
}

template<class Pc>
void ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::write_to_stream( std::ostream &os ) const {
    os << "pos: ";
    for( int i = 0; i < nb_nodes(); ++i )
        os << ( i ? " [" : "[" ) << nodes.xs[ i ] << " " << nodes.ys[ i ] << "]";
}

template<class Pc> template<class F>
void ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::for_each_bound( const F &/*f*/ ) const {
    TODO;
}

template<class Pc> template<class F>
void ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::for_each_node( const F &f ) const {
    for( int i = 0; i < nb_nodes(); ++i )
        f( node( i ) );
}

template<class Pc>
int ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::nb_nodes() const {
    return nodes_size;
}

template<class Pc>
bool ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::empty() const {
    return nodes_size == 0;
}

template<class Pc>
typename ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::Pt ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::node( int index ) const {
    return { nodes.xs[ index ], nodes.ys[ index ] };
}

template<class Pc> template<int flags,class Fu>
void ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags>, const Fu &fu ) {
    //    constexpr int ss = SimdSize<TF>::value;
    // Prop: on fait une version <= 8 noeuds, ou on stocke tout dans des registres pour le plane cut.
    // Si ça dépasse, on teste avec <= 64

    // max 8 nodes version (we store coords in registers)
    std::size_t num_cut = 0;
    if ( nodes_size <= 8 ) {
        using LF = SimdVec<TF,8>;
        using LC = SimdVec<CI,8>;

        // load in registers
        LF px = LF::load_aligned( nodes.xs );
        LF py = LF::load_aligned( nodes.ys );
        LC pc = LC::load_aligned( nodes.cut_ids );

        for( ; num_cut < nb_cuts; ++num_cut ) {
            // get distance and outside bit for each node
            TF cx = cut_dir[ 0 ][ num_cut ];
            TF cy = cut_dir[ 1 ][ num_cut ];
            TF cs = cut_ps[ num_cut ];

            LF bi = px * LF( cx ) + py * LF( cy );
            std::uint16_t outside_nodes = bi > LF( cs );

            // if nothing has changed => go to the next cut
            if ( outside_nodes == 0 )
                continue;

            //
            std::uint16_t nmsk = 1 << nodes_size;
            std::uint16_t case_code = ( outside_nodes & ( nmsk - 1 ) ) | nmsk;
            LF di = bi - LF( cs );

            //
            #include "(ConvexPolyhedron2dLT64_plane_cut_switch.cpp).h"

            TODO;
        }

        // save from registers
        LF::store_aligned( nodes.xs, px );
        LF::store_aligned( nodes.ys, py );
        LC::store_aligned( nodes.cut_ids, pc );
    }

    // => data does not fit in registers
    for( ; num_cut < nb_cuts; ++num_cut ) {
        TODO;
        //        // get distance and outside bit for each node
        //        TF cx = cut_dir[ 0 ][ num_cut ];
        //        TF cy = cut_dir[ 1 ][ num_cut ];
        //        TF cs = cut_ps[ num_cut ];
        //        std::uint64_t outside_nodes = 0;
        //        SimdRange<ss>::for_each( nodes_size, [&]( int n, auto s ) {
        //            using LF = SimdVec<TF,s.val>;
        //            using LC = SimdVec<CI,s.val>;
        //            LF px = LF::load_aligned( nodes.xs + n );
        //            LF py = LF::load_aligned( nodes.ys + n );
        //            LC pc = LC::load_aligned( nodes.ys + n );

        //            LF bi = px * LF( cx ) + py * LF( cy );
        //            LF::store_aligned( nodes.ds + n, bi - LF( cs ) );
        //            std::uint64_t lo = bi > LF( cs );
        //            outside_nodes |= lo << n;
        //        } );

        //        // if nothing has changed => go to the next cut
        //        if ( outside_nodes == 0 )
        //            continue;

        //        //
        //        #include "(ConvexPolyhedron2dLt64_plane_cut_switch.cpp).h"
    }

    fu( *this );
}

template<class Pc>
void ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ) {
    return plane_cut( cut_dir, cut_ps, cut_id, nb_cuts, N<0>(), [&]( auto & ) {} );
}

template<class Pc> template<class TL>
void ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>::Bound::foreach_simplex( const TL &/*f*/ ) const {
    // f( Simplex<TF,2,1>{ points[ 0 ], points[ 1 ] } );
    TODO;
}

} // namespace sdot
