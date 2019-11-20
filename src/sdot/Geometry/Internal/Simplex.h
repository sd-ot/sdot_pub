#ifndef SDOT_SIMPLEX_H
#define SDOT_SIMPLEX_H

#include "../../Support/PointTraits.h"
#include <functional>

namespace sdot {

/**
*/
template<class TF,int dim,int sub_dim=dim>
struct Simplex;


template<class TF>
struct Simplex<TF,3,2> {
    using Pt = Point3<TF>;

    TF    mass        () const { return 0.5 * norm_2( cross_prod( pts[ 1 ] - pts[ 0 ], pts[ 2 ] - pts[ 0 ] ) ); }
    Pt    random_point( const std::function<TF()> &rand_func ) const { TF u = rand_func(), v = rand_func(); if ( u + v > 1 ) { u = TF( 1 ) - u; v = TF( 1 ) - v; }; return ( TF( 1 ) - u - v ) * pts[ 0 ] + u * pts[ 1 ] + v * pts[ 2 ]; }

    Pt    pts[ 3 ];
};

} // namespace sdot

#endif // SDOT_SIMPLEX_H