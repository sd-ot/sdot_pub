#include "ConvexPolyhedron3Gen.h"
#include "../Support/TODO.h"

namespace sdot {

template<class Pc>
ConvexPolyhedron3Gen<Pc>::ConvexPolyhedron3Gen() {
    num_cut_proc = 0;
}

template<class Pc>
ConvexPolyhedron3Gen<Pc> &ConvexPolyhedron3Gen<Pc>::operator=( const ConvexPolyhedron3Lt64<Pc> &cp ) {
    nodes.clear();
    faces.clear();

    cp.for_each_node( [&]( const auto &node ) {
        nodes.push_back( { node.pos() } );
    } );

    cp.for_each_face( [&]( const auto &face ) {
        Face new_face;
        new_face.nodes.reserve( face.nb_nodes() );
        face.for_each_node_index_sec( [&]( int num_node ) {
            new_face.nodes.push_back( num_node );
        } );
        faces.push_back( std::move( new_face ) );
    } );

    return *this;
}

template<class Pc>
void ConvexPolyhedron3Gen<Pc>::display_vtk( VtkOutput &vo, const std::vector<TF> &cell_values, Pt offset, bool /*display_both_sides*/ ) const {
    std::vector<VtkOutput::Pt> pts;
    pts.reserve( 16 );
    for_each_face( [&]( const Face &face ) {
        pts.clear();
        for( int n : face.nodes )
            pts.push_back( nodes[ n ].pos() + offset );
        vo.add_polygon( pts, cell_values );
    } );
}


template<class Pc>
void ConvexPolyhedron3Gen<Pc>::for_each_boundary_item( const std::function<void( const BoundaryItem &boundary_item )> &/*f*/ ) const {
    TODO;
}

template<class Pc>
void ConvexPolyhedron3Gen<Pc>::for_each_face( const std::function<void( const Face &face )> &f ) const {
    for( const Face &face : faces )
        f( face );
}

template<class Pc>
void ConvexPolyhedron3Gen<Pc>::for_each_node( const std::function<void( const Node &node )> &f ) const {
    TODO;
}

template<class Pc> template<int flags>
void ConvexPolyhedron3Gen<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ) {
    for( std::size_t num_cut = 0; num_cut < nb_cuts; ++num_cut ) {
        // node_dists
        new_nodes.clear();
        bool all_inside = true;
        node_repls.resize( nb_nodes() );
        node_dists.resize( nb_nodes() );
        for( std::size_t i = 0; i < nb_nodes(); ++i ) {
            TF ps = nodes[ i ].p[ 0 ] * cut_dir[ 0 ][ num_cut ] +
                    nodes[ i ].p[ 1 ] * cut_dir[ 1 ][ num_cut ] +
                    nodes[ i ].p[ 2 ] * cut_dir[ 2 ][ num_cut ] ;
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

        // faces
        new_faces.clear();
        for( const Face &face : faces ) {
            Face new_face;
            for( std::size_t i0 = 0; i0 < face.nodes.size(); ++i0 ) {
                int n0 = face.nodes[ i0 ];
                if ( node_dists[ n0 ] > 0 )
                    continue;

                int nm = face.nodes[ ( i0 + face.nodes.size() - 1 ) % face.nodes.size() ];
                if ( node_dists[ nm ] > 0 ) {
                }

                new_face.nodes.push_back( node_repls[ n0 ] );

                int n1 = face.nodes[ ( i0 + 1 ) % face.nodes.size() ];
                if ( node_dists[ n1 ] > 0 ) {
                }
            }
            new_faces.push_back( std::move( new_face ) );
        }

        //
        std::swap( nodes, new_nodes );
        std::swap( faces, new_faces );
    }
}

template<class Pc>
void ConvexPolyhedron3Gen<Pc>::plane_cut( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ) {
    plane_cut( cut_dir, cut_ps, cut_id, nb_cuts, N<0>() );
}


} // namespace sdot
