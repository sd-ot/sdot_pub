#include "ConvexPolyhedron3dVoid.h"
#include "../../Support/TODO.h"
#include "../../Support/P.h"

namespace sdot {

template<class Pc>
ConvexPolyhedron<Pc,3,void>::ConvexPolyhedron( Pt pmin, Pt pmax, CI cut_id ) : ConvexPolyhedron() {
    nodes = {
        { .p = { pmin[ 0 ], pmin[ 1 ], pmin[ 2 ] } },
        { .p = { pmax[ 0 ], pmin[ 1 ], pmin[ 2 ] } },
        { .p = { pmin[ 0 ], pmax[ 1 ], pmin[ 2 ] } },
        { .p = { pmax[ 0 ], pmax[ 1 ], pmin[ 2 ] } },
        { .p = { pmin[ 0 ], pmin[ 1 ], pmax[ 2 ] } },
        { .p = { pmax[ 0 ], pmin[ 1 ], pmax[ 2 ] } },
        { .p = { pmin[ 0 ], pmax[ 1 ], pmax[ 2 ] } },
        { .p = { pmax[ 0 ], pmax[ 1 ], pmax[ 2 ] } },
    };

    faces = {
        { .nodes = { 0, 2, 3, 1 }, .cut_id = cut_id, .normal = { 0, 0, -1 } },
        { .nodes = { 4, 5, 7, 6 }, .cut_id = cut_id, .normal = { 0, 0, +1 } },
        { .nodes = { 0, 1, 5, 4 }, .cut_id = cut_id, .normal = { 0, -1, 0 } },
        { .nodes = { 2, 6, 7, 3 }, .cut_id = cut_id, .normal = { 0, +1, 0 } },
        { .nodes = { 0, 4, 6, 2 }, .cut_id = cut_id, .normal = { -1, 0, 0 } },
        { .nodes = { 1, 3, 7, 5 }, .cut_id = cut_id, .normal = { +1, 0, 0 } },
    };
}

template<class Pc>
ConvexPolyhedron<Pc,3,void>::ConvexPolyhedron() {
    num_cut_proc = 0;
}

template<class Pc>
void ConvexPolyhedron<Pc,3,void>::write_to_stream( std::ostream &os ) const {
    os << "Nodes:";
    for( const Node &node : nodes ) {
        os << "\n ";
        for( TF v : node.p )
            os << " " << v;
    }

    os << "\nFaces:";
    for( const Face &face : faces ) {
        os << "\n ";
        for( int index : face.nodes )
            os << " " << index;

        os << " normal:";
        for( TF v : face.normal )
            os << " " << v;
    }
}

template<class Pc>
void ConvexPolyhedron<Pc,3,void>::for_each_boundary_item( const std::function<void( const BoundaryItem &boundary_item )> &f ) const {
    for( const Face &face : faces )
        f( { .cp = this, .face = &face } );
}

template<class Pc>
void ConvexPolyhedron<Pc,3,void>::for_each_node( const std::function<void( const Pt &p )> &f ) const {
    for( const Node &node : nodes )
        f( node.p );
}

template<class Pc> template<int flags,class Fu>
void ConvexPolyhedron<Pc,3,void>::plane_cut( std::array<const TF *,Pc::dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags>, const Fu & ) {
    for( std::size_t num_cut = 0; num_cut < nb_cuts; ++num_cut ) {
        // node_dists
        new_nodes.clear();
        bool all_inside = true;
        node_repls.resize( nb_nodes() );
        node_dists.resize( nb_nodes() );
        for( std::size_t i = 0; i < nb_nodes(); ++i ) {
            TF ps = 0;
            for( int d = 0; d < Pc::dim; ++d )
                ps += nodes[ i ].p[ d ] * cut_dir[ d ][ num_cut ];
            node_dists[ i ] = ps - cut_ps[ num_cut ];
            node_repls[ i ] = new_nodes.size();

            if ( node_dists[ i ] > 0 )
                all_inside = false;
            else
                new_nodes.push_back( nodes[ i ] );
        }

        // no cut ?
        if ( all_inside )
            return;

        // we need this to determine if we already have created a node on an edge*
        std::size_t nb_edges = nb_nodes() * ( nb_nodes() + 1 ) / 2;
        num_cut_proc_edge.resize( nb_edges, 0 );
        num_node_edge.resize( nb_edges );
        ++num_cut_proc;

        // linked list(s) for the new face(s)
        std::size_t old_prev_cut_node_size = new_nodes.size();
        prev_cut_node.resize( new_nodes.size() );

        // faces
        new_faces.clear();
        for( const Face &face : faces ) {
            Face new_face;
            new_face.cut_id = face.cut_id;

            // helper
            auto add_node_between = [&]( int num_node_0, int num_node_1 ) {
                int min_node_128 = std::min( num_node_0, num_node_1 );
                int max_node_128 = std::max( num_node_0, num_node_1 );
                int num_edge_128 = max_node_128 * ( max_node_128 + 1 ) / 2 + min_node_128;

                int num_node_128;
                if ( num_cut_proc_edge[ num_edge_128 ] != num_cut_proc ) {
                    const Pt &n0 = nodes[ num_node_0 ].p;
                    const Pt &n1 = nodes[ num_node_1 ].p;
                    TF d0 = node_dists[ num_node_0 ];
                    TF d1 = node_dists[ num_node_1 ];

                    num_node_128 = new_nodes.size();
                    num_node_edge[ num_edge_128 ] = new_nodes.size();
                    num_cut_proc_edge[ num_edge_128 ] = num_cut_proc;
                    new_nodes.push_back( { n0 + d0 / ( d0 - d1 ) * ( n1 - n0 ) } );
                    prev_cut_node.push_back( -1 );
                } else
                    num_node_128 = num_node_edge[ num_edge_128 ];

                new_face.nodes.push_back( num_node_128 );
                return num_node_128;
            };

            // look up for a node after an outside one
            std::size_t o0 = 0;
            while ( o0 < face.nodes.size() && ! ( node_dists[ face.nodes[ o0 ] ] > 0 ) )
                ++o0;
            ++o0;

            // for each inside node
            int prev_out_in;
            for( std::size_t i0 = o0; i0 < o0 + face.nodes.size(); ++i0 ) {
                int n0 = face.nodes[ ( i0 + 0 ) % face.nodes.size() ];
                if ( node_dists[ n0 ] > 0 )
                    continue;

                int nm = face.nodes[ ( i0 - 1 ) % face.nodes.size() ];
                if ( node_dists[ nm ] > 0 )
                    prev_out_in = add_node_between( nm, n0 );

                new_face.nodes.push_back( node_repls[ n0 ] );

                int n1 = face.nodes[ ( i0 + 1 ) % face.nodes.size() ];
                if ( node_dists[ n1 ] > 0 ) {
                    int nn = add_node_between( n0, n1 );
                    prev_cut_node[ prev_out_in ] = nn;
                }
            }

            // store
            new_faces.push_back( std::move( new_face ) );
        }

        // new face(s)
        if ( new_faces.size() ) {
            for( std::size_t num = old_prev_cut_node_size; num < new_nodes.size(); ++num ) {
                int nxt = prev_cut_node[ num ];
                if ( nxt < 0 )
                    continue;

                Face new_face;
                new_face.cut_id = cut_id[ num_cut ];
                new_face.nodes.push_back( num );
                prev_cut_node[ num ] = -1;
                do {
                    new_face.nodes.push_back( nxt );
                    int old = prev_cut_node[ nxt ];
                    prev_cut_node[ nxt ] = -1;
                    nxt = old;
                } while ( nxt >= 0 );

                new_faces.push_back( std::move( new_face ) );
            }
        }


        //
        std::swap( nodes, new_nodes );
        std::swap( faces, new_faces );
    }
}

template<class Pc>
void ConvexPolyhedron<Pc,3,void>::plane_cut( std::array<const TF *,Pc::dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ) {
    plane_cut( cut_dir, cut_ps, cut_id, nb_cuts, N<0>(), []() {} );
}


template<class Pc>
void ConvexPolyhedron<Pc,3,void>::BoundaryItem::for_each_node( const std::function<void(Pt)> &f ) const {
    for( int n : face->nodes )
        f( cp->nodes[ n ].p );
}

} // namespace sdot
