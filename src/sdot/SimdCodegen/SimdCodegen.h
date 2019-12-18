#pragma once

#include "Path.h"
#include "Lane.h"

/**
*/
class SimdCodegen {
public:
    /**/ SimdCodegen  ( int simd_size = 4 );

    Path best_path_for( Reg out, std::vector<Lane> res );

    int  simd_size;
};


