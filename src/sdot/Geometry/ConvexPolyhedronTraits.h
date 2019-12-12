#pragma once

#include "ConvexPolyhedron2.h"
#include "ConvexPolyhedron3.h"

namespace sdot {

template<class Pc,int dim=Pc::dim> struct ConvexPolyhedronTraits {};

template<class Pc> struct ConvexPolyhedronTraits<Pc,2> { using type = ConvexPolyhedron2<Pc>; };
template<class Pc> struct ConvexPolyhedronTraits<Pc,3> { using type = ConvexPolyhedron3Lt64<Pc>; };

}
