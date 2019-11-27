#pragma once

#include "CellBoundsTraits.h"

namespace sdot {
template<class Pc> struct LGridOutOfCoreCell;
template<class Pc> struct LGridSuperCell;
template<class Pc> struct LGridFinalCell;

/**
*/
template<class Pc>
struct LGridBaseCell {
    using            CellBounds         = typename CellBoundsTraits<Pc>::type;
    using            SuperCell          = LGridSuperCell<Pc>;
    using            FinalCell          = LGridFinalCell<Pc>;
    using            OOOCell            = LGridOutOfCoreCell<Pc>;
    using            TI                 = typename Pc::TI;

    const OOOCell   *out_of_core_cell   () const { return nb_sub_items == 0 ? static_cast<const OOOCell *>( this ) : nullptr; }
    OOOCell         *out_of_core_cell   () { return nb_sub_items == 0 ? static_cast<OOOCell *>( this ) : nullptr; }

    const SuperCell *super_cell         () const { return nb_sub_items < 0 ? static_cast<const SuperCell *>( this ) : nullptr; }
    SuperCell       *super_cell         () { return nb_sub_items < 0 ? static_cast<SuperCell *>( this ) : nullptr; }

    const FinalCell *final_cell         () const { return nb_sub_items > 0 ? static_cast<const FinalCell *>( this ) : nullptr; }
    FinalCell       *final_cell         () { return nb_sub_items > 0 ? static_cast<FinalCell *>( this ) : nullptr; }

    TI               end_ind_in_fcells; ///< end index in final cells
    int              nb_sub_items;      ///< > 0 => final cell (nb diracs). < 0 => super cell (nb sub cells).
    CellBounds       bounds;            ///< pos and weight bounds
};

} // namespace sdot
