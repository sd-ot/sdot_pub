#pragma once

#include "../../Support/BumpPointerPool.h"
#include "LGridBaseCell.h"

namespace sdot {

/**
*/
template<class Pc>
struct LGridFinalCell : LGridBaseCell<Pc> {
    using                FinalCell     = LGridFinalCell<Pc>;
    using                BaseCell      = LGridBaseCell<Pc>;
    using                Dirac         = typename Pc::Dirac;

    /**/                ~LGridFinalCell();

    std::size_t          size_in_bytes () const;
    int                  nb_diracs     () const { return this->nb_sub_items; }
    static FinalCell    *allocate      ( BumpPointerPool &pool, std::size_t &ram_acc, int nb_diracs );
    FinalCell          *&next          ();

    Dirac                diracs[ 1 ];  ///<
};

template<class Pc>
LGridFinalCell<Pc>::~LGridFinalCell() {
    for( int i = 1; i < nb_diracs(); ++i )
        diracs[ i ].~Dirac();
}

template<class Pc>
LGridFinalCell<Pc> *LGridFinalCell<Pc>::allocate( BumpPointerPool &pool, std::size_t &ram_acc, int nb_diracs ) {
    std::size_t ram = sizeof( BaseCell ) + nb_diracs * sizeof( Dirac );
    FinalCell *res = new ( pool.allocate( ram + sizeof( LGridFinalCell ) ) ) FinalCell;
    for( int i = 1; i < nb_diracs; ++i )
        new ( res->diracs + i ) Dirac;
    res->nb_sub_items = nb_diracs;
    res->next() = nullptr;
    ram_acc += ram;
    return res;
}

template<class Pc>
std::size_t LGridFinalCell<Pc>::size_in_bytes() const {
    return sizeof( BaseCell ) + nb_diracs() * sizeof( Dirac );
}

template<class Pc>
LGridFinalCell<Pc> *&LGridFinalCell<Pc>::next() {
    char *b = reinterpret_cast<char *>( this );
    return *reinterpret_cast<LGridFinalCell<Pc> **>( b + sizeof( BaseCell ) + nb_diracs() * sizeof( Dirac ) );
}

} // namespace sdot
