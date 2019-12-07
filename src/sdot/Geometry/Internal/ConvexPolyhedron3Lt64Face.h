#pragma once

#include "ConvexPolyhedron3Lt64NodeBlock.h"
#include "ConvexPolyhedron3Lt64FaceBlock.h"
#include <functional>

namespace sdot {
template<class Pc>
class ConvexPolyhedron3;

/**
  Data layout:
*/
template<class Carac>
class alignas( 64 ) ConvexPolyhedron3Lt64Face {
public:
    using     Node               = ConvexPolyhedron3Lt64NodeBlock<Carac>;
    using     Cp                 = ConvexPolyhedron3<Carac>;

    void      write_to_stream    ( std::ostream &/*os*/ ) const { /*os << cp->faces.node_lists[ num_face ];*/ }

    void      for_each_node_index( const std::function<void( int index )> &f ) const;
    void      for_each_node      ( const std::function<void( const Node &node )> &f ) const;

    int       num_face;
    const Cp *cp;
};

template<class Carac>
void ConvexPolyhedron3Lt64Face<Carac>::for_each_node_index( const std::function<void(int)> &f ) const {
    for( int i = 0; i < cp->faces.nb_nodes[ num_face ]; ++i )
        f( cp->faces.node_lists[ num_face ][ i ] );
}

template<class Carac>
void ConvexPolyhedron3Lt64Face<Carac>::for_each_node( const std::function<void (const ConvexPolyhedron3Lt64Face::Node &)> &f ) const {
    for( int num_node = 0; num_node < cp->faces.nb_nodes[ num_face ]; ++num_node )
        f( cp->node( cp->faces.node_lists[ num_face ][ num_node ] ) );
}

} // namespace sdot

