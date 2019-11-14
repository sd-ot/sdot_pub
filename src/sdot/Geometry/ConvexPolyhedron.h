#pragma once

#include "../Support/VtkOutput.h"
#include "../Support/N.h"

namespace sdot {

/**
*/
class ConvexPolyhedron {
public:
    // flags for plane cut
    static constexpr int plane_cut_flag_dir_is_normalized = 1;
    static constexpr int do_not_use_switch                = 2;
    static constexpr int do_not_use_simd                  = 4;
};

}
