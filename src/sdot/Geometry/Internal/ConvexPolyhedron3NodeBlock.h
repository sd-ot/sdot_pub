#pragma once

#include "PaddedType.h"
#include "../Point3.h"

namespace sdot {

/**
  Data layout:
*/
template<class TF,class TI,int bs>
class alignas(32) ConvexPolyhedron3NodeBlock {
public:
    using       Node                     = ConvexPolyhedron3NodeBlock;
    using       PI                       = PaddedType<int,bs,sizeof(TF),(sizeof(int)>sizeof(TF))>;
    using       Pt                       = Point3<TF>;

    Pt          pos                      () const { return { x, y, z }; }

    const Node& local_at                 ( TI index ) const { return *reinterpret_cast<const Node *>( &x + index ); }
    Node&       local_at                 ( TI index ) { return *reinterpret_cast<Node *>( &x + index ); }

    const Node& global_at                ( TI index ) const { return *reinterpret_cast<const Node *>( &this[ index / bs ].x + index % bs ); }
    Node&       global_at                ( TI index ) { return *reinterpret_cast<Node *>( &this[ index / bs ].x + index % bs ); }

    void        get_content_from         ( const Node &b ) { get_straight_content_from( b ); }
    void        get_straight_content_from( const Node &b ) { x = b.x; y = b.y; z = b.z; }

    TF          x, _pad_x[ bs - 1 ];
    TF          y, _pad_y[ bs - 1 ];
    TF          z, _pad_z[ bs - 1 ];
};

} // namespace sdot

