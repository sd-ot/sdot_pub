#pragma once

#include "LGridBaseCell.h"

namespace sdot {

/**
*/
template<class Pc>
struct LGridParentCell : LGridBaseCell<Pc> {
    int                nb_sub_cells () const { return 1 - this->nb_sub_items; }
    std::size_t        size_in_bytes() const { return sizeof( LGridBaseCell<Pc> ) + nb_sub_cells() * sizeof( LGridBaseCell<Pc> * ); }

    LGridBaseCell<Pc> *sub_cells[ 4 ]; ///<
};

} // namespace sdot
