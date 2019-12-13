#pragma once

#include "Internal/ConvexPolyhedronFlags.h"

namespace sdot {

///  Pc must contain
///    - CI => cut info
///    - TF (double, ...) => floating point type
///    - dim => ...
///
///  Opt => optimisation. void means "generic case".
///
///  This version is generic, and far from optimized.
template<class Pc,int dim=Pc::dim,class Opt=void>
class ConvexPolyhedron;

} // namespace sdot

#include "Internal/ConvexPolyhedron3dVoid.h"

// specializations
