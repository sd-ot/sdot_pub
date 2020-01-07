#pragma once

#include "../../Support/Point.h"

namespace sdot {

/**
  Data layout:
*/
template<class TF,class CI,int bs,bool store_the_normals>
class alignas(64) ConvexPolyhedron2dLt64_NodeBlock {
public:
    using       Node   = ConvexPolyhedron2dLt64_NodeBlock;
    using       Pt     = Point<TF,2>;

    Pt          pos    ( int n ) const { return { xs[ n ], ys[ n ] }; }
    Pt          dir    ( int n ) const { return { dir_xs[ n ], dir_ys[ n ] }; }

    void        cpy    ( int dst, int src ) { xs[ dst ] = xs[ src ]; ys[ dst ] = ys[ src ]; if ( store_the_normals ) { dir_xs[ dst ] = dir_xs[ src ]; dir_ys[ dst ] = dir_ys[ src ]; } cut_ids[ dst ] = cut_ids[ src ]; }

    TF          xs     [ bs ];  ///<
    TF          ys     [ bs ];  ///<
    TF          dir_xs [ bs ];  ///<
    TF          dir_ys [ bs ];  ///<
    CI          cut_ids[ bs ];  ///<
};

} // namespace sdot

