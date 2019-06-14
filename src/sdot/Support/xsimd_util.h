#pragma once

#include <xsimd/xsimd.hpp>


inline std::uint64_t is_neg( xsimd::batch<double,4> val ) {
    return _mm256_movemask_pd( val < 0 );
}

inline std::uint64_t is_neg( xsimd::batch<double,8> val ) {
    return val < 0;
}
