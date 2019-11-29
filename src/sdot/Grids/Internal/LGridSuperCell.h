#pragma once

#include "../../Support/BumpPointerPool.h"
#include "LGridBaseCell.h"

namespace sdot {

/**
*/
template<class Pc>
struct LGridSuperCell : LGridBaseCell<Pc> {
    std::size_t            size_in_bytes() const { return sizeof( LGridBaseCell<Pc> ) + nb_sub_cells() * sizeof( LGridBaseCell<Pc> * ); }
    int                    nb_sub_cells () const { return - this->nb_sub_items; }
    static LGridSuperCell *allocate     ( BumpPointerPool &pool, std::size_t &ram, int nb_sub_cells ); ///< nb_sub_cells must be > 1 (because 1 corresponds to a OutOfCoreCell)

    LGridBaseCell<Pc>     *sub_cells    [ 1 << Pc::dim ]; ///<
};

// impl ------------------------------------------------------------------------------------
template<class Pc>
LGridSuperCell<Pc> *LGridSuperCell<Pc>::allocate( BumpPointerPool &pool, std::size_t &ram_acc, int nb_sub_cells ) {
    std::size_t ram = sizeof( LGridBaseCell<Pc> ) + nb_sub_cells * sizeof( LGridBaseCell<Pc> * );
    LGridSuperCell *res = new ( pool.allocate( ram ) ) LGridSuperCell;
    res->nb_sub_items = - nb_sub_cells;
    ram_acc += ram;
    return res;
}

} // namespace sdot
