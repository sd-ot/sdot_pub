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
    using                NodeList              = std::array<std::int8_t,max_nb_nodes_per_face>;
    using                NodeMask              = std::uint64_t;
    using                CI                    = typename Carac::Dirac *;
    static constexpr int bs                    = max_nb_faces_per_cell;
    using                TF                    = typename Carac::TF;

    NodeMask             node_masks[ bs ];     ///< per face
    NodeList             node_lists[ bs ];     ///< per face
    TF                   normal_xs [ bs ];     ///< per face
    TF                   normal_ys [ bs ];     ///< per face
    TF                   normal_zs [ bs ];     ///< per face
    unsigned             nb_nodes  [ bs ];     ///< per face
    CI                   cut_ids   [ bs ];     ///< per face

    unsigned             tmp      [ bs ];     ///< per face
};

} // namespace sdot

