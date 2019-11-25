#pragma once

#include <cstdint>
#include <bitset>
#include <array>

namespace sdot {

/**
  Data layout:
*/
template<class Carac>
class alignas(32) ConvexPolyhedron3Lt64FaceBlock {
public:
    using                NodeList = std::array<std::uint8_t,8>;
    using                NodeMask = std::bitset<64>;
    using                Face     = ConvexPolyhedron3Lt64FaceBlock;
    using                TF       = typename Carac::TF;
    using                TI       = typename Carac::TI;
    static constexpr int bs       = 256;

    const Face&          local_at ( TI index ) const { return *reinterpret_cast<const Face *>( &node_mask + index ); }
    Face&                local_at ( TI index ) { return *reinterpret_cast<Face *>( &node_mask + index ); }

    const Face&          global_at( TI index ) const { return *reinterpret_cast<const Face *>( &this[ index / bs ].x + index % bs ); }
    Face&                global_at( TI index ) { return *reinterpret_cast<Face *>( &this[ index / bs ].x + index % bs ); }

    NodeMask             node_mask, _node_masks[ bs - 1 ];
    NodeList             node_list, _node_lists[ bs - 1 ];
    TF                   normal_x , _normal_x  [ bs - 1 ];
    TF                   normal_y , _normal_y  [ bs - 1 ];
    TF                   normal_z , _normal_z  [ bs - 1 ];
};

} // namespace sdot

