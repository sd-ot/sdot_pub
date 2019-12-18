#pragma once

#include "Path.h"
#include <array>

/**
*/
class SimdCodegen {
public:
    /**/ SimdCodegen  ( int simd_size = 4 );

    Path best_path_for( std::vector<int> res );

    int  simd_size;
};


