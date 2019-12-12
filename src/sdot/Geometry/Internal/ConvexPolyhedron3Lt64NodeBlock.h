#pragma once

#include "../../Support/Point3.h"

namespace sdot {

/**
  Data layout:
*/
template<class Carac>
class ConvexPolyhedron3Lt64NodeBlock {
public:
    // common types
    using       TF                       = typename Carac::TF;
    using       Pt                       = Point3<TF>;
    enum {      nb                       = 64 }; ///< nb items
    enum {      bs                       = 4 * nb }; ///< block size. 1 (old size) + 2 (max nb additionnal nodes: 3 edges per node...) + 1 (tmp nodes, to be moved afterwards)

    // shortcuts
    using       Node                     = ConvexPolyhedron3Lt64NodeBlock;

    // methods
    void        set_pos                  ( Pt p ) { x = p.x; y = p.y; z = p.z; }
    Pt          pos                      () const { return { x, y, z }; }

    const Node& local_at                 ( int index ) const { return *reinterpret_cast<const Node *>( &x + index ); }
    Node&       local_at                 ( int index ) { return *reinterpret_cast<Node *>( &x + index ); }

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

