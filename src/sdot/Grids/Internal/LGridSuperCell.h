#pragma  once

#include "LGridBaseCell.h"

namespace sdot {

/**
*/
template<class Pc>
struct LGridSuperCell : LGridBaseCell<Pc> {
    using     BaseCell        = LGridBaseCell<Pc>;

    int       nb_sub_cells    () const { return - this->nb_sub_items; }

    BaseCell *sub_cells[ 1 ]; ///<
};

} // namespace sdot
