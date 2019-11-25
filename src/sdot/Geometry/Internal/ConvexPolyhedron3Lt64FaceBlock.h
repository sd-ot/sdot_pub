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
    using                NodeLstp          = std::array<std::int8_t,8>;
    using                NodeMask          = std::bitset<64>;
    using                Face              = ConvexPolyhedron3Lt64FaceBlock;
    using                TF                = typename Carac::TF;
    using                TI                = typename Carac::TI;
    static constexpr int bs                = 256;

    const Face&          local_at          ( TI index ) const { return *reinterpret_cast<const Face *>( &node_mask + index ); }
    Face&                local_at          ( TI index ) { return *reinterpret_cast<Face *>( &node_mask + index ); }

    const Face&          global_at         ( TI index ) const { return *reinterpret_cast<const Face *>( &this[ index / bs ].x + index % bs ); }
    Face&                global_at         ( TI index ) { return *reinterpret_cast<Face *>( &this[ index / bs ].x + index % bs ); }

    template<class F>
    void                 foreach_node_index( const F &f ) const {
        for( std::size_t i = 0; i < 8; ++i ) {
            if ( node_lst0[ i ] < 0 )
                return;
            f( node_lst0[ i ] );
        }
        for( std::size_t i = 0; i < 8; ++i ) {
            if ( node_lst1[ i ] < 0 )
                return;
            f( node_lst1[ i ] );
        }
    }

    NodeMask             node_mask, _node_masks[ bs - 1 ];
    NodeLstp             node_lst0, _node_lst0s[ bs - 1 ];
    NodeLstp             node_lst1, _node_lst1s[ bs - 1 ];
    TF                   normal_x , _normal_x  [ bs - 1 ];
    TF                   normal_y , _normal_y  [ bs - 1 ];
    TF                   normal_z , _normal_z  [ bs - 1 ];
};

} // namespace sdot

