#include "CellBoundsPpos.h"

namespace sdot {

template<class Pc>
void CellBoundsPpos<Pc>::LocalSolver::clr() {
    mat_weight.setZero();
    vec_weight.setZero();

    min_pos = + std::numeric_limits<TF>::max();
    max_pos = - std::numeric_limits<TF>::max();

    n = 0;
}

template<class Pc>
void CellBoundsPpos<Pc>::LocalSolver::push( Pt pos, TF weight ) {
    using std::max;
    using std::min;

    std::array<TF,nb_coeffs_w_bound> shape_functions;
    shape_functions[ 0 ] = 1;
    if ( w_bounds_order >= 1 )
        for( std::size_t d = 0; d < dim; ++d )
            shape_functions[ d + 1 ] = pos[ d ];
    for( std::size_t r = 0; r < nb_coeffs_w_bound; ++r ) {
        for( std::size_t c = 0; c < nb_coeffs_w_bound; ++c )
            mat_weight( r, c ) += shape_functions[ r ] * shape_functions[ c ];
        vec_weight( r ) += shape_functions[ r ] * weight;
    }

    max_pos = max( max_pos, pos );
    min_pos = min( min_pos, pos );

    ++n;
}

template<class Pc>
void CellBoundsPpos<Pc>::LocalSolver::push( const LocalSolver &ls ) {
    using std::max;
    using std::min;

    mat_weight += ls.mat_weight;
    vec_weight += ls.vec_weight;

    max_pos = max( max_pos, ls.max_pos );
    min_pos = min( min_pos, ls.min_pos );

    n += ls.n;
}

template<class Pc>
void CellBoundsPpos<Pc>::LocalSolver::store_to( CellBoundsPpos &bounds ) {
    using std::max;
    using std::abs;

    TF m = 0;
    for( std::size_t i = 0; i < nb_coeffs_w_bound; ++i )
        m = max( m, abs( mat_weight.coeff( i, i ) ) );
    m *= 1e-10;
    for( std::size_t i = 0; i < nb_coeffs_w_bound; ++i )
        mat_weight.coeffRef( i, i ) += m;

    Eigen::LLT<TMat> llt;
    llt.compute( mat_weight );
    TVec res = llt.solve( vec_weight );
    for( std::size_t i = 0; i < res.size(); ++i )
        bounds.poly_weight[ i ] = res[ i ];

    bounds.min_pos = min_pos;
    bounds.max_pos = max_pos;
}

template<class Pc>
typename CellBoundsPpos<Pc>::TF CellBoundsPpos<Pc>::get_w( Pt pos ) const {
    TF res = poly_weight[ 0 ];
    for( std::size_t d = 0; d < dim; ++d )
        res += poly_weight[ d + 1 ] * pos[ d ];
    return res;
}


template<class Pc>
void CellBoundsPpos<Pc>::push( Pt pos, TF weight) {
    using std::max;

    TF res = weight;
    for( std::size_t d = 0; d < dim; ++d )
        res -= poly_weight[ d + 1 ] * pos[ d ];
    poly_weight[ 0 ] = max( poly_weight[ 0 ], res );
}

} // namespace sdot
