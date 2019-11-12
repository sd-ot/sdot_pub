#pragma once

#include "CellBoundsPpos.h"
#include "CellBoundsP0.h"

namespace sdot {

template<class Pc,bool order_pos=(Pc::w_bounds_order>0)> struct CellBoundsTraits {};

template<class Pc> struct CellBoundsTraits<Pc,0> { using type = CellBoundsP0  <Pc>; };
template<class Pc> struct CellBoundsTraits<Pc,1> { using type = CellBoundsPpos<Pc>; };

}
