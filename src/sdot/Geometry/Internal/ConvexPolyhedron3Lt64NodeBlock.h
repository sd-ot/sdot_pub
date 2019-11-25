#pragma once

#include "../../Support/Point3.h"
#include "PaddedType.h"

namespace sdot {

/**
  Data layout:
*/
template<class Carac>
class alignas(32) ConvexPolyhedron3Lt64NodeBlock {
public:
    // common types
    using       TF                       = typename Carac::TF;
    using       TI                       = typename Carac::TI;
    using       Pt                       = Point3<TF>;
    enum {      bs                       = 64 }; ///< block size

    // content used if edge is round
    struct      RoundStuff               {
        Pt      tangent_0;               ///< tangent in n0
        Pt      tangent_1;               ///< tangent in n1
        TF      angle_1;                 ///< angle of n1 (angle of n0 = 0)
        TF      radius;                  ///<
        Pt      center;                  ///<
        Pt      ndir;                    ///< normalized( node[ n0 ].pos - center )
    };

    // shortcuts
    using       Node                     = ConvexPolyhedron3Lt64NodeBlock;

    // methods
    void        set_pos                  ( Pt p ) { x = p.x; y = p.y; z = p.z; }
    Pt          pos                      () const { return { x, y, z }; }

    const Node& local_at                 ( TI index ) const { return *reinterpret_cast<const Node *>( &x + index ); }
    Node&       local_at                 ( TI index ) { return *reinterpret_cast<Node *>( &x + index ); }

    const Node& global_at                ( TI index ) const { return *reinterpret_cast<const Node *>( &this[ index / bs ].x + index % bs ); }
    Node&       global_at                ( TI index ) { return *reinterpret_cast<Node *>( &this[ index / bs ].x + index % bs ); }

    bool        outside                  () const { return d > 0; }
    bool        inside                   () const { return ! outside(); }

    void        get_content_from         ( const Node &b ) { get_straight_content_from( b ); }
    void        get_straight_content_from( const Node &b ) { x = b.x; y = b.y; z = b.z; }

    // attributes
    TF          x, _pad_x[ bs - 1 ];     ///< position
    TF          y, _pad_y[ bs - 1 ];     ///< position
    TF          z, _pad_z[ bs - 1 ];     ///< position
    TF          d, _pad_d[ bs - 1 ];     ///< dist, computed for each intersection
};

} // namespace sdot

