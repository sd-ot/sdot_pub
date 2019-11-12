#ifndef SDOT_CellBoundsP0_H
#define SDOT_CellBoundsP0_H

#include "../../Geometry/PointTraits.h"

namespace sdot {

/**
*/
template<class Pc>
class CellBoundsP0 {
public:
    static constexpr bool        need_phase_1      = false;
    using                        LocalSolver       = CellBoundsP0<Pc>;
    static constexpr std::size_t dim               = Pc::dim;            ///<
    using                        TF                = typename Pc::TF;
    using                        Pt                = typename PointTraits<TF,dim>::type;

    void                         clr               ();
    void                         push              ( Pt pos, TF weight );
    void                         push              ( const LocalSolver &ls );
    void                         store_to          ( CellBoundsP0 &bounds );

    TF                           get_w             ( Pt pos ) const;

    TF                           max_weight;
    Pt                           min_pos;
    Pt                           max_pos;
};

} // namespace sdot

#include "CellBoundsP0.tcc"

#endif // SDOT_CellBoundsP0_H
