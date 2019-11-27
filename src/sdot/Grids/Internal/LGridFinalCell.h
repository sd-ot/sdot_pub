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
        std::size_t ram = sizeof( BaseCell ) + nb_diracs * sizeof( Dirac );
        FinalCell *res = new ( mem_pool.allocate( ram ) ) FinalCell;
        res->nb_sub_items = nb_diracs;
        res->ram = ram;
        return res;
    }

    int               nb_diracs    () const { return this->nb_sub_items; }

    Dirac             diracs[ 1 ]; ///<
};

} // namespace sdot
