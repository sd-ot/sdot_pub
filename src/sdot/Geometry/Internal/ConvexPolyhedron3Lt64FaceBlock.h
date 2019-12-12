#pragma once

#include <cstdint>
#include <array>

namespace sdot {

/**
  Data layout:
*/
template<class Carac>
class alignas( 64 ) ConvexPolyhedron3Lt64FaceBlock {
public:
    static constexpr int max_nb_faces_per_cell = 256;
    static constexpr int max_nb_nodes_per_face = 16;
    static constexpr int max_nb_nodes_per_cell = 64;
    using                NodeList              = std::array<std::uint8_t,max_nb_nodes_per_face>;
    using                NodeMask              = std::uint64_t;
    using                CI                    = typename Carac::Dirac *;
    static constexpr int bs                    = max_nb_faces_per_cell;
    using                TF                    = typename Carac::TF;

    void                 cpy                   ( int dst, int src );

    NodeMask             node_masks[ bs ];     ///< per face
    NodeList             node_lists[ bs ];     ///< per face
    TF                   normal_xs [ bs ];     ///< per face
    TF                   normal_ys [ bs ];     ///< per face
    TF                   normal_zs [ bs ];     ///< per face
    int                  off_in_ans[ bs ];     ///< per face. Offset in additional_nums. Used for nodes >= max_nb_nodes_per_face
    int                  nb_nodes  [ bs ];     ///< per face
    CI                   cut_ids   [ bs ];     ///< per face
};

// --------------------------------------------------------------------------------------------
template<class Carac>
void ConvexPolyhedron3Lt64FaceBlock<Carac>::cpy( int dst, int src ) {
    node_masks[ dst ] = node_masks[ src ];
    node_lists[ dst ] = node_lists[ src ];
    normal_xs [ dst ] = normal_xs [ src ];
    normal_ys [ dst ] = normal_ys [ src ];
    normal_zs [ dst ] = normal_zs [ src ];
    nb_nodes  [ dst ] = nb_nodes  [ src ];
    cut_ids   [ dst ] = cut_ids   [ src ];
}

} // namespace sdot

