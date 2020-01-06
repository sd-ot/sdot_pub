#pragma once

namespace sdot {

///  Pc must contain
///    - CI => cut info
///    - TF (double, ...) => floating point type
///    - dim => space dimensionnality (if not specified in template args)
///
///  Variant => void means "generic case". Other types may be defined later
///
///  This version is generic, and far from optimized.
template<class Pc,int dim=Pc::dim,class Variant=void>
class ConvexPolyhedron;

} // namespace sdot

#include "Internal/ConvexPolyhedron_2d.h"
// #include "Internal/ConvexPolyhedron_3d.h"

// specializations
