#pragma once

#include "PaddedType.h"

namespace sdot {

/**
  Data layout:
*/
template<class TF,class TI,int bs,bool allow_ball_cut>
class alignas(32) ConvexPolyhedron3EdgeBlock {
public:
    using       Edge                     = ConvexPolyhedron3EdgeBlock;
    using       PI                       = PaddedType<int,bs,sizeof(TF),(sizeof(int)>sizeof(TF))>;

    int         num_node                 ( int o ) { return o ? node_1.get() : node_0.get(); }

    void        set_face                 ( int o, TI num ) { ( o ? face_1 : face_0 ).set( num ); }

    const Edge& local_at                 ( TI index ) const { return *reinterpret_cast<const Edge *>( reinterpret_cast<const TF *>( this ) + index ); }
    Edge&       local_at                 ( TI index ) { return *reinterpret_cast<Edge *>( reinterpret_cast<const TF *>( this ) + index ); }

    const Edge& global_at                ( TI index ) const { return *reinterpret_cast<const Edge *>( reinterpret_cast<const TF *>( this + index / bs ) + index % bs ); }
    Edge&       global_at                ( TI index ) { return *reinterpret_cast<Edge *>( reinterpret_cast<const TF *>( this + index / bs ) + index % bs ); }

    void        get_content_from         ( const Edge &b ) { get_straight_content_from( b ); if ( allow_ball_cut ) {
                                                             tangent_0_x = b.tangent_0_x; tangent_0_y = b.tangent_0_y; tangent_0_z = b.tangent_0_z;
                                                             tangent_1_x = b.tangent_1_x; tangent_1_y = b.tangent_1_y; tangent_1_z = b.tangent_1_z;
                                                             center_x    = b.center_x   ; center_y    = b.center_y   ; center_z    = b.center_z   ;
                                                             angle_1     = b.angle_1    ; radius      = b.radius     ; } }

    void        get_straight_content_from( const Edge &b ) { node_0 = b.node_0; node_1 = b.node_1; face_0 = b.face_0; face_1 = b.face_1; ndir_x = b.ndir_x; ndir_y = b.ndir_y; ndir_z = b.ndir_z; }

    PI          node_0;
    PI          node_1;

    PI          face_0;
    PI          face_1;

    TF          ndir_x      , _pad_ndir_x      [ bs - 1 ]; ///< normalized( node[ n0 ].pos - center )
    TF          ndir_y      , _pad_ndir_y      [ bs - 1 ]; ///< normalized( node[ n0 ].pos - center )
    TF          ndir_z      , _pad_ndir_z      [ bs - 1 ]; ///< normalized( node[ n0 ].pos - center )

    TF          tangent_0_x , _pad_tangent_0_x [ bs - 1 ]; ///< tangent in n0
    TF          tangent_0_y , _pad_tangent_0_y [ bs - 1 ]; ///< tangent in n0
    TF          tangent_0_z , _pad_tangent_0_z [ bs - 1 ]; ///< tangent in n0

    TF          tangent_1_x , _pad_tangent_1_x [ bs - 1 ]; ///< tangent in n1
    TF          tangent_1_y , _pad_tangent_1_y [ bs - 1 ]; ///< tangent in n1
    TF          tangent_1_z , _pad_tangent_1_z [ bs - 1 ]; ///< tangent in n1

    TF          center_x    , _pad_center_x    [ bs - 1 ]; ///<
    TF          center_y    , _pad_center_y    [ bs - 1 ]; ///<
    TF          center_z    , _pad_center_z    [ bs - 1 ]; ///<

    TF          angle_1     , _pad_angle_1     [ bs - 1 ]; ///< angle of n1 (angle of n0 = 0)

    TF          radius      , _pad_radius      [ bs - 1 ]; ///<
};

} // namespace sdot

