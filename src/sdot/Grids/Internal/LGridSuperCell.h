#pragma once

#include "../../Support/BumpPointerPool.h"
#include "LGridBaseCell.h"

namespace sdot {

/**
*/
template<class Pc>
struct LGridSuperCell : LGridBaseCell<Pc> {
    using             SuperCell   = LGridSuperCell<Pc>;
    using             BaseCell    = LGridBaseCell<Pc>;
    
    static SuperCell *allocate    ( BumpPointerPool &mem_pool, int nb_sub_cells ) {
        std::size_t ram = sizeof( BaseCell ) + nb_sub_cells * sizeof( BaseCell * );
        SuperCell *res = new ( mem_pool.allocate( ram ) ) SuperCell;
        res->nb_sub_items = - nb_sub_cells;
        res->ram = ram;
        return res;
    }

    int               nb_sub_cells() const { return - this->nb_sub_items; }
                     
    BaseCell         *sub_cells[ 1 ]; ///<
};

} // namespace sdot
