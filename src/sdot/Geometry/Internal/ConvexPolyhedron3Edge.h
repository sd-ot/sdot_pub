#pragma once

#include "../../Support/PointerWithSmallOffset.h"
#include "../Point3.h"

namespace sdot {

template<class Carac> class ConvexPolyhedron3NodeBlock;
template<class Carac> class ConvexPolyhedron3Edge;
template<class Carac> class ConvexPolyhedron3Face;

/**
  Data layout:
*/
template<class Carac>
class ConvexPolyhedron3Edge {
public:
    using       Content              = PointerWithSmallOffset<ConvexPolyhedron3NodeBlock<Carac>,4>;
    using       Node                 = ConvexPolyhedron3NodeBlock<Carac>;
    using       Edge                 = ConvexPolyhedron3Edge<Carac>;
    using       Face                 = ConvexPolyhedron3Face<Carac>;
    using       TF                   = typename Carac::TF;
    using       TI                   = typename Carac::TI;
    using       Pt                   = Point3<TF>;

    /**/        ConvexPolyhedron3Edge( const ConvexPolyhedron3Edge &that ) : content( that.content ) {}
    /**/        ConvexPolyhedron3Edge( Node *n0, int o0 = 0 ) : content( n0, o0 ) {}
    /**/        ConvexPolyhedron3Edge() {}

    Edge        next                 () const { return content.ptr()->next_in_faces[ content.offset() ].get(); }
    Node       *n0                   () const { return content.ptr(); }
    Node       *n1                   () const { return next().n0(); }

    operator    bool                 () const { return content; }

private:
    Content     content;
};

} // namespace sdot

