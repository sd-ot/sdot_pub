#pragma once

#include "../../Support/BumpPointerPool.h"
#include "LGridBaseCell.h"

namespace sdot {

/**
*/
template<class Pc>
struct LGridFinalCell : LGridBaseCell<Pc> {
    using             FinalCell    = LGridFinalCell<Pc>;
    using             BaseCell     = LGridBaseCell<Pc>;
    using             Dirac        = typename Pc::Dirac;

    static FinalCell *allocate     ( BumpPointerPool &mem_pool, int nb_diracs ) {
        FinalCell *res = new ( mem_pool.allocate( sizeof( BaseCell ) + nb_diracs * sizeof( Dirac ) ) ) FinalCell;
        res->nb_sub_items = nb_diracs;
        return res;
    }

    int               nb_diracs    () const { return this->nb_sub_items; }

    Dirac             diracs[ 1 ]; ///<
};

} // namespace sdot
