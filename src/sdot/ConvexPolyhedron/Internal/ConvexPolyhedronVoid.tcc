#include "ConvexPolyhedronVoid.h"
#include "../../Support/TODO.h"
#include "../../Support/P.h"

namespace sdot {

template<class Pc>
ConvexPolyhedron<Pc,void>::ConvexPolyhedron( Pt pmin, Pt pmax, CI cut_id ) : ConvexPolyhedron() {
    construct_rec( faces, nodes, pmin, pmax, cut_id, N<dim>() );
}

template<class Pc>
ConvexPolyhedron<Pc,void>::ConvexPolyhedron() {
    num_cut_proc = 0;
}

template<class Pc> template<int d>
void ConvexPolyhedron<Pc,void>::construct_rec( std::vector<Face> &faces, std::vector<Node> &nodes, Pt pmin, Pt pmax, CI cut_id, N<d> ) {
    // d - 1
    std::vector<Face> s_faces;
    std::vector<Node> s_nodes;
    construct_rec( s_faces, s_nodes, pmin, pmax, cut_id, N<d-1>() );

    // make the nodes
    for( std::size_t i = 0; i < s_nodes.size(); ++i ) {
        s_nodes[ i ].p[ d - 1 ] = pmin[ d - 1 ];
        nodes.push_back( { .p = s_nodes[ i ].p } );
    }
    for( std::size_t i = s_nodes.size(); i--; ) {
        s_nodes[ i ].p[ d - 1 ] = pmax[ d - 1 ];
        nodes.push_back( { .p = s_nodes[ i ].p } );
    }

    // faces
    Face face;
    face.normal = TF( 0 );
    face.cut_id = cut_id;

    face.normal[ d - 1 ] = -1;
    for( std::size_t i = 0; i < s_nodes.size(); ++i )
        face.nodes.push_back( i );
    faces.push_back( std::move( face ) );

    face.nodes.clear();
    face.normal[ d - 1 ] = +1;
    for( std::size_t i = 0; i < s_nodes.size(); ++i )
        face.nodes.push_back( s_nodes.size() + i );
    faces.push_back( std::move( face ) );

    for( const Face &s_face : s_faces ) {
        face.nodes.clear();
        face.normal = s_face.normal;

        if ( faces.size() % 2 ) {
            for( int n : s_face.nodes )
                face.nodes.push_back( n );
            for( std::size_t i = s_face.nodes.size(); i--; )
                face.nodes.push_back( nodes.size() - 1 - s_face.nodes[ i ] );
        } else {
            for( std::size_t i = s_face.nodes.size(); i--; )
                face.nodes.push_back( nodes.size() - 1 - s_face.nodes[ i ] );
            for( int n : s_face.nodes )
                face.nodes.push_back( n );
        }

        faces.push_back( std::move( face ) );
    }
}

template<class Pc>
void ConvexPolyhedron<Pc,void>::construct_rec( std::vector<Face> &, std::vector<Node> &nodes, Pt, Pt, CI, N<0> ) {
    nodes.emplace_back();
}

template<class Pc>
void ConvexPolyhedron<Pc,void>::write_to_stream( std::ostream &os ) const {
    os << "Nodes:";
    for_each_node( [&]( const Pt &p ) {
        os << "\n ";
        for( TF v : p )
            os << " " << v;
    } );

    os << "\nFaces:";
    for_each_face( [&]( const auto &face ) {
        os << "\n ";
        face.for_each_node_index( [&]( int index ) {
            os << " " << index;
        } );

        os << " normal:";
        for( TF v : face.normal )
            os << " " << v;

        // P( face.normal, cross_prod( nodes[ face.nodes[ 3 ] ].p - nodes[ face.nodes[ 0 ] ].p, nodes[ face.nodes[ 1 ] ].p - nodes[ face.nodes[ 0 ] ].p ) );
    } );

}

template<class Pc>
void ConvexPolyhedron<Pc,void>::for_each_boundary_item( const std::function<void( const BoundaryItem &boundary_item )> &/*f*/ ) const {
    TODO;
}

template<class Pc>
void ConvexPolyhedron<Pc,void>::for_each_face( const std::function<void( const Face &face )> &f ) const {
    for( const Face &face : faces )
        f( face );
}

template<class Pc>
void ConvexPolyhedron<Pc,void>::for_each_node( const std::function<void( const Pt &p )> &f ) const {
    for( const Node &node : nodes )
        f( node.p );
}

template<class Pc> template<int flags>
void ConvexPolyhedron<Pc,void>::plane_cut( std::array<const TF *,Pc::dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ) {
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
            while ( ! ( node_dists[ face.nodes[ o0 ] ] > 0 ) )
                if ( ++o0 == face.nodes.size() )
                    break;
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
                new_face.cut_id = cut_id[ num ];
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
void ConvexPolyhedron<Pc,void>::plane_cut( std::array<const TF *,Pc::dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ) {
    plane_cut( cut_dir, cut_ps, cut_id, nb_cuts, N<0>() );
}


} // namespace sdot
