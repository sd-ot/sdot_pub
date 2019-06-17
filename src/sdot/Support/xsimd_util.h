#pragma once

#include <xsimd/xsimd.hpp>


#ifdef __AVX2__
inline std::uint64_t is_neg( xsimd::batch<double,4> val ) {
    return _mm256_movemask_pd( val < 0 );
}
#endif

#ifdef __AVX512F__
inline std::uint64_t is_neg( xsimd::batch<double,8> val ) {
    return val < 0;
}
#endif
