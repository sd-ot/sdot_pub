#pragma once

#include "../../Support/BumpPointerPool.h"
#include "CellBoundsTraits.h"

namespace sdot {

/**
*/
template<class Pc>
struct LGridFinalCell {
    using                  CellBounds    = typename CellBoundsTraits<Pc>::type;
    using                  Dirac         = typename Pc::Dirac;
    using                  TI            = typename Pc::TI;

    /**/                  ~LGridFinalCell();

    std::size_t            size_in_bytes () const;
    static LGridFinalCell *allocate      ( BumpPointerPool &pool, std::size_t &ram_acc, int nb_diracs );

    TI                     end_ind_in_fcells; ///< end index in final cells
    CellBounds             bounds;            ///< pos and weight bounds
    int                    nb_diracs;         ///< > 0 => final cell (nb diracs). < 0 => super cell (nb sub cells).
    Dirac                  diracs[ 1 ];       ///<
};

template<class Pc>
LGridFinalCell<Pc>::~LGridFinalCell() {
    for( int i = 1; i < nb_diracs; ++i )
        diracs[ i ].~Dirac();
}

template<class Pc>
LGridFinalCell<Pc> *LGridFinalCell<Pc>::allocate( BumpPointerPool &pool, std::size_t &ram_acc, int nb_diracs ) {
    std::size_t ram = sizeof( LGridFinalCell ) + ( nb_diracs - 1 ) * sizeof( Dirac );
    LGridFinalCell *res = new ( pool.allocate( ram + sizeof( LGridFinalCell ) ) ) LGridFinalCell;
    for( int i = 1; i < nb_diracs; ++i )
        new ( res->diracs + i ) Dirac;
    res->nb_sub_items = nb_diracs;
    res->next() = nullptr;
    ram_acc += ram;
    return res;
}

template<class Pc>
std::size_t LGridFinalCell<Pc>::size_in_bytes() const {
    return sizeof( LGridFinalCell ) + ( nb_diracs - 1 ) * sizeof( Dirac );
}

} // namespace sdot
