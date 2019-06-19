#pragma once

#define _USE_MATH_DEFINES
// #include <math.h>
#include <cmath>
#include "S.h"

namespace sdot {

template<class TF>
inline TF pi( S<TF> ) {
    using std::atan;
    return 4 * atan( TF( 1 ) );
}

inline double pi( S<double> ) {
    return M_PI;
}

inline float pi( S<float> ) {
    return M_PI;
}

}
