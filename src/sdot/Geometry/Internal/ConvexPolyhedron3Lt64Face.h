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
    using     Node           = ConvexPolyhedron3Lt64NodeBlock<Carac>;
    using     Cp             = ConvexPolyhedron3<Carac>;

    void      write_to_stream( std::ostream &/*os*/ ) const { /*os << cp->faces.node_lists[ num_face ];*/ }
    void      for_each_node  ( const std::function<void( const Node &node )> &f ) const { for( unsigned num_node = 0; num_node < cp->faces.nb_nodes[ num_face ]; ++num_node ) f( cp->node( cp->faces.node_lists[ num_face ][ num_node ] ) ); }

    unsigned  num_face;
    const Cp *cp;
};

} // namespace sdot

