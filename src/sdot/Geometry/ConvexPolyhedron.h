#pragma once

#include "../Support/N.h"
#include "../VtkOutput.h"

/**
*/
class ConvexPolyhedron {
public:
    // flags for plane cut
    static constexpr int plane_cur_flag_dir_is_normalized = 1;
    static constexpr int plane_cut_flag_no_switches       = 2;
    static constexpr int do_not_use_simd                  = 4;
};
