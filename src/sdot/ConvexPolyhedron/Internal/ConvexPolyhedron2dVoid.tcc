#include "../../Support/Simplex.h"
#include "../../Support/ASSERT.h"
#include "../../Support/TODO.h"
#include "../../Support/P.h"
#include "ConvexPolyhedron2dVoid.h"

namespace sdot {

template<class Pc>
ConvexPolyhedron<Pc,2>::ConvexPolyhedron( Pt pmin, Pt pmax, CI cut_id ) : ConvexPolyhedron() {
    nodes = {
        { .p = { pmin[ 0 ], pmin[ 1 ] }, .n = {  0, -1 }, .cut_id = cut_id },
        { .p = { pmax[ 0 ], pmin[ 1 ] }, .n = { +1,  0 }, .cut_id = cut_id },
        { .p = { pmax[ 0 ], pmax[ 1 ] }, .n = {  0, +1 }, .cut_id = cut_id },
        { .p = { pmin[ 0 ], pmax[ 1 ] }, .n = { -1,  0 }, .cut_id = cut_id },
    };
}

template<class Pc>
ConvexPolyhedron<Pc,2>::ConvexPolyhedron() {
}

template<class Pc>
void ConvexPolyhedron<Pc,2>::write_to_stream( std::ostream &os ) const {
    for( int i = 0; i < nb_nodes(); ++i )
        os << ( i ? " [" : "[" ) << node( i )[ 0 ] << " " << node( i )[ 1 ] << "]";
}

template<class Pc> template<class F>
void ConvexPolyhedron<Pc,2>::for_each_bound( const F &f ) const {
    for( std::size_t n0 = nodes.size() - 1, n1 = 0; n1 < nodes.size(); n0 = n1++ )
        f( Bound{ .n0 = nodes.data() + n0, .n1 = nodes.data() + n1 } );
}

template<class Pc> template<class F>
void ConvexPolyhedron<Pc,2>::for_each_node( const F &f ) const {
    for( int i = 0; i < nb_nodes(); ++i )
        f( node( i ) );
}

template<class Pc>
int ConvexPolyhedron<Pc,2>::nb_nodes() const {
    return nodes.size();
}

template<class Pc>
bool ConvexPolyhedron<Pc,2>::empty() const {
    return nodes.empty();
}

template<class Pc>
typename ConvexPolyhedron<Pc,2>::Pt ConvexPolyhedron<Pc,2>::node( int index ) const {
    return nodes[ index ].p;
}

template<class Pc> template<int flags,class Fu>
void ConvexPolyhedron<Pc,2>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags>, const Fu &fu ) {
    // => more than 8 nodes, but less than 65
    for( std::size_t num_cut = 0; num_cut < nb_cuts; ++num_cut ) {
        TF cx = cut_dir[ 0 ][ num_cut ];
        TF cy = cut_dir[ 1 ][ num_cut ];
        TF cs = cut_ps[ num_cut ];

        // get distance for each node
        int nb_outside = 0;
        for( Node &node : nodes ) {
            node.d = node.p[ 0 ] * cx + node.p[ 1 ] * cy - cs;
            nb_outside += node.outside();
        }

        // if nothing has changed => go to the next cut
        if ( nb_outside == 0 )
            continue;

        // make a new edge set, in a tmp storage
        new_nodes.clear();
        for( std::size_t n0 = 0, nm = nodes.size() - 1; n0 < nodes.size(); nm = n0++ ) {
            if ( nodes[ n0 ].outside() )
                continue;

            if ( nodes[ nm ].outside() ) {
                TF m = nodes[ n0 ].d / ( nodes[ nm ].d - nodes[ n0 ].d );
                new_nodes.push_back( {
                    .p = {
                        nodes[ n0 ].p[ 0 ] - m * ( nodes[ nm ].p[ 0 ] - nodes[ n0 ].p[ 0 ] ),
                        nodes[ n0 ].p[ 1 ] - m * ( nodes[ nm ].p[ 1 ] - nodes[ n0 ].p[ 1 ] )
                    },
                    .cut_id = nodes[ nm ].cut_id
                } );
            }

            new_nodes.push_back( nodes[ n0 ] );

            int n1 = ( n0 + 1 ) % nodes.size();
            if ( nodes[ n1 ].outside() ) {
                TF m = nodes[ n0 ].d / ( nodes[ n1 ].d - nodes[ n0 ].d );
                new_nodes.push_back( {
                    .p = {
                        nodes[ n0 ].p[ 0 ] - m * ( nodes[ n1 ].p[ 0 ] - nodes[ n0 ].p[ 0 ] ),
                        nodes[ n0 ].p[ 1 ] - m * ( nodes[ n1 ].p[ 1 ] - nodes[ n0 ].p[ 1 ] )
                    },
                    .cut_id = cut_id[ num_cut ]
                } );
            }
        }

        //
        std::swap( nodes, new_nodes );
    }

    fu( *this );
}

template<class Pc>
void ConvexPolyhedron<Pc,2>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ) {
    return plane_cut( cut_dir, cut_ps, cut_id, nb_cuts, N<0>(), [&]( auto & ) {} );
}

template<class Pc> template<class TL>
void ConvexPolyhedron<Pc,2>::Bound::for_each_simplex( const TL &f ) const {
    f( Simplex<TF,2,1>{ n0->p, n1->p } );
}

template<class Pc>
typename Pc::CI ConvexPolyhedron<Pc,2>::Bound::cut_id() const {
    return n0->cut_id;
}

} // namespace sdot
