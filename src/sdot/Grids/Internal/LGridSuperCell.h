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
        SuperCell *res = reinterpret_cast<SuperCell *>( mem_pool.allocate( sizeof( BaseCell ) + nb_sub_cells * sizeof( BaseCell * ) ) );
        res->nb_sub_items = - nb_sub_cells;
        return res;
    }

    int               nb_sub_cells() const { return - this->nb_sub_items; }
                     
    BaseCell         *sub_cells[ 1 ]; ///<
};

} // namespace sdot
