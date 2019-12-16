#pragma once

#include "Internal/ConvexPolyhedronFlags.h"
#include "../Support/Point.h"
#include "../Support/N.h"
#include <functional>

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

#include "Internal/ConvexPolyhedron2dVoid.h"
#include "Internal/ConvexPolyhedron2dLt64.h"

#include "Internal/ConvexPolyhedron3dVoid.h"
#include "Internal/ConvexPolyhedron3dLt64.h"

// specializations
