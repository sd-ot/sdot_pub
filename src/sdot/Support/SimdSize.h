#pragma once

#include <x86intrin.h>

namespace sdot {

/**
  Best size for a given type
*/
template<class TF> struct SimdSize { enum { value = 1 }; };

#if defined(__AVX512F__)
    template<> struct SimdSize<double> { enum { value = 8 }; };
#elif defined(__AVX2__)
    template<> struct SimdSize<double> { enum { value = 4 }; };
#endif


} // namespace sdot
