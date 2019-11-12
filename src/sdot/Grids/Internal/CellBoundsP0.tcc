#include "CellBoundsP0.h"

namespace sdot {

template<class Pc>
void CellBoundsP0<Pc>::clr() {
    max_weight = - std::numeric_limits<TF>::max();
    min_pos    = + std::numeric_limits<TF>::max();
    max_pos    = - std::numeric_limits<TF>::max();
}

template<class Pc>
void CellBoundsP0<Pc>::push( const CellBoundsP0::LocalSolver &ls ) {
    using std::max;
    using std::min;

    max_weight = max( max_weight, ls.max_weight );
    max_pos    = max( max_pos   , ls.max_pos    );
    min_pos    = min( min_pos   , ls.min_pos    );
}

template<class Pc>
void CellBoundsP0<Pc>::push( CellBoundsP0::Pt pos, CellBoundsP0::TF weight ) {
    using std::max;
    using std::min;

    max_weight = max( max_weight, weight );
    max_pos    = max( max_pos   , pos    );
    min_pos    = min( min_pos   , pos    );
}

template<class Pc>
void CellBoundsP0<Pc>::store_to( CellBoundsP0 &bounds ) {
    bounds.max_weight = max_weight;
    bounds.min_pos    = min_pos;
    bounds.max_pos    = max_pos;
}

template<class Pc>
typename CellBoundsP0<Pc>::TF CellBoundsP0<Pc>::get_w( CellBoundsP0::Pt pos ) const {
    return max_weight;
}

} // namespace sdot
