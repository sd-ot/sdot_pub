#pragma once

#include "../../Support/Point2.h"
#include "PaddedType.h"

namespace sdot {

/**
  Data layout:
*/
template<class TF,class TI,class CI,int bs,bool store_the_normals,bool allow_ball_cut>
class alignas(32) ConvexPolyhedron2NodeBlock {
public:
    using       Node                     = ConvexPolyhedron2NodeBlock;
    using       PCI                      = PaddedType<CI,bs,sizeof(TF),(sizeof(CI )>sizeof(TF))>;
    using       Pt                       = Point2<TF>;

    Pt          pos                      () const { return { x, y }; }
    Pt          dir                      () const { return { dir_x, dir_y }; }

    const Node& local_at                 ( TI index ) const { return *reinterpret_cast<const Node *>( &x + index ); }
    Node&       local_at                 ( TI index ) { return *reinterpret_cast<Node *>( &x + index ); }

    const Node& global_at                ( TI index ) const { return *reinterpret_cast<const Node *>( &this[ index / bs ].x + index % bs ); }
    Node&       global_at                ( TI index ) { return *reinterpret_cast<Node *>( &this[ index / bs ].x + index % bs ); }

    void        get_content_from         ( const Node &b ) { get_straight_content_from( b ); if ( allow_ball_cut ) { arc_radius = b.arc_radius; arc_center_x = b.arc_center_x; arc_center_y = b.arc_center_y; } }
    void        get_straight_content_from( const Node &b ) { x = b.x; y = b.y; if ( store_the_normals ) { dir_x = b.dir_x; dir_y = b.dir_y; } cut_id.set( b.cut_id.get() ); }

    TF          x           , _pad_x           [ bs - 1 ];
    TF          y           , _pad_y           [ bs - 1 ];
    TF          dir_x       , _pad_dir_x       [ bs - 1 ];
    TF          dir_y       , _pad_dir_y       [ bs - 1 ];
    TF          arc_radius  , _pad_arc_radius  [ bs - 1 ]; // < 0 if straight line
    TF          arc_center_x, _pad_arc_center_x[ bs - 1 ]; // arc center x
    TF          arc_center_y, _pad_arc_center_y[ bs - 1 ]; // arc center y
    PCI         cut_id;                                    // actually stored as TI
};

} // namespace sdot

