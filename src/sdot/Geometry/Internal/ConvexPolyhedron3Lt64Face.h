#pragma once

#include "ConvexPolyhedron3Lt64NodeBlock.h"
#include "ConvexPolyhedron3Lt64FaceBlock.h"
#include <functional>

namespace sdot {
template<class Pc>
class ConvexPolyhedron3Lt64;

/**
  Data layout:
*/
template<class Carac>
class alignas( 64 ) ConvexPolyhedron3Lt64Face {
public:
    using     Node                   = ConvexPolyhedron3Lt64NodeBlock<Carac>;
    using     Cp                     = ConvexPolyhedron3Lt64<Carac>;

    void      write_to_stream        ( std::ostream &/*os*/ ) const { /*os << cp->faces.node_lists[ num_face ];*/ }

    void      for_each_node_index_sec( const std::function<void( int index )> &f ) const; ///< version that works even if nb_fnodes > 16
    void      for_each_node_index    ( const std::function<void( int index )> &f ) const;
    void      for_each_node          ( const std::function<void( const Node &node )> &f ) const;
    int       nb_nodes               () const { return cp->faces.nb_nodes[ num_face ]; }

    int       num_face;
    const Cp *cp;
};

template<class Carac>
void ConvexPolyhedron3Lt64Face<Carac>::for_each_node_index_sec( const std::function<void(int)> &f ) const {
    constexpr int mn = ConvexPolyhedron3Lt64FaceBlock<Carac>::max_nb_nodes_per_face;
    if ( nb_nodes() <= mn )
        return for_each_node_index( f );

    for( int i = 0; i < mn; ++i )
        f( cp->faces.node_lists[ num_face ][ i ] );
    for( int i = mn; i < nb_nodes(); ++i )
        f( cp->additional_nums[ cp->faces.off_in_ans[ num_face ] + i - mn ] );
}

template<class Carac>
void ConvexPolyhedron3Lt64Face<Carac>::for_each_node_index( const std::function<void(int)> &f ) const {
    for( int i = 0; i < nb_nodes(); ++i )
        f( cp->faces.node_lists[ num_face ][ i ] );
}

template<class Carac>
void ConvexPolyhedron3Lt64Face<Carac>::for_each_node( const std::function<void (const ConvexPolyhedron3Lt64Face::Node &)> &f ) const {
    for( int num_node = 0; num_node < nb_nodes(); ++num_node )
        f( cp->node( cp->faces.node_lists[ num_face ][ num_node ] ) );
}

} // namespace sdot

