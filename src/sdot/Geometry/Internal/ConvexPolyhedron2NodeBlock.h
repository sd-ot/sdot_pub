#pragma once

#include "ConvexPolyhedron2NodeCutId.h"
#include "../Point2.h"

namespace sdot {

/**
  Data layout:
*/
template<class TF,class TI,TI bs>
class ConvexPolyhedron2NodeBlock {
public:
    using Node     = ConvexPolyhedron2NodeBlock;
    using CI       = ConvexPolyhedron2NodeCutId<TI,bs,sizeof(TF)/sizeof(TI),sizeof(TI)/sizeof(TF)>;
    using Pt       = Point2<TF>;

    Pt    pos      () const { return { x, y }; }
    Pt    dir      () const { return { dir_x, dir_y }; }

    Node& local_at ( TI index ) { return *reinterpret_cast<Node *>( &x + index ); }
    Node& global_at( TI index ) { return *reinterpret_cast<Node *>( &this[ index / bs ].x + index % bs ); }

    TF    x           , _pad_x           [ bs - 1 ];
    TF    y           , _pad_y           [ bs - 1 ];
    TF    dir_x       , _pad_dir_x       [ bs - 1 ];
    TF    dir_y       , _pad_dir_y       [ bs - 1 ];
    TF    arc_radius  , _pad_arc_radius  [ bs - 1 ]; // < 0 if straight line
    TF    arc_center_x, _pad_arc_center_x[ bs - 1 ]; // arc center x
    TF    arc_center_y, _pad_arc_center_y[ bs - 1 ]; // arc center y
    CI    cut_id;                                    // actually stored as TI
};

} // namespace sdot

