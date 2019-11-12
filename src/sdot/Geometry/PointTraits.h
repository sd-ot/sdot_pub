#pragma once

#include "Point1.h"
#include "Point2.h"
#include "Point3.h"

namespace sdot {

template<class TF,int dim> struct PointTraits {};

template<class TF> struct PointTraits<TF,1> { using type = Point1<TF>; };
template<class TF> struct PointTraits<TF,2> { using type = Point2<TF>; };
template<class TF> struct PointTraits<TF,3> { using type = Point3<TF>; };

}
