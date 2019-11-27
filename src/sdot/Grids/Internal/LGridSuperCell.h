#pragma once

#include "../../Support/BumpPointerPool.h"
#include "LGridParentCell.h"

namespace sdot {

/**
*/
template<class Pc>
struct LGridSuperCell : LGridParentCell<Pc> {
    static LGridSuperCell *allocate( BumpPointerPool &pool, std::size_t &ram, int nb_sub_cells ); ///< nb_sub_cells must be > 1 (because 1 corresponds to a OutOfCoreCell)
};

// impl ------------------------------------------------------------------------------------
template<class Pc>
LGridSuperCell<Pc> *LGridSuperCell<Pc>::allocate( BumpPointerPool &pool, std::size_t &ram_acc, int nb_sub_cells ) {
    std::size_t ram = sizeof( LGridBaseCell<Pc> ) + nb_sub_cells * sizeof( LGridBaseCell<Pc> * );
    LGridSuperCell *res = new ( pool.allocate( ram ) ) LGridSuperCell;
    res->nb_sub_items = 1 - nb_sub_cells;
    ram_acc += ram;
    return res;
}

} // namespace sdot
