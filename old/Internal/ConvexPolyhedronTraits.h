#pragma once

#include "ConvexPolyhedron2D.h"
#include "ConvexPolyhedron3.h"

namespace sdot {

template<class Pc,int dim=Pc::dim> struct ConvexPolyhedronTraits {};

template<class Pc> struct ConvexPolyhedronTraits<Pc,2> { using type = ConvexPolyhedron2D<Pc>; };
template<class Pc> struct ConvexPolyhedronTraits<Pc,3> { using type = ConvexPolyhedron3Lt64<Pc>; };

}
