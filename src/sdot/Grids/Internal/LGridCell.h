#pragma once

#include "../../Support/Span.h"
#include "LGridFinalCell.h"

namespace sdot {

/**
*/
template<class Pc>
struct LGridSuperCell {
    using                     CellBounds   = typename CellBoundsTraits<Pc>::type;
    using                     TI           = typename Pc::TI;
    union                     Child        { LGridFinalCell<Pc> *fcell; };
    union                     Data         { Child children[ 1 ]; };

    std::size_t               size_in_bytes() const { return sizeof( LGridSuperCell ) + ( nb_scells + nb_fcells + nb_ocells - 1 ) * sizeof( void * ); }
    static LGridSuperCell*    allocate     ( BumpPointerPool &pool, std::size_t &ram, int nb_fcells, int nb_scells, int nb_ocells ); ///<

    LGridFinalCell<Pc>*&      fcell        ( int index ) { return _fcells[ index ]; }
    LGridSuperCell*&          scell        ( int index ) { return reinterpret_cast<LGridSuperCell **>( _fcells + nb_fcells )[ index ]; }
    std::size_t&              ocell        ( int index ) { return reinterpret_cast<std::size_t *>( _fcells + nb_fcells + nb_scells )[ index ]; }

    Span<LGridFinalCell<Pc>*> fcells       () { return { _fcells, nb_fcells }; }
    Span<LGridSuperCell*>     scells       () { return { reinterpret_cast<LGridSuperCell **>( _fcells + nb_fcells ), nb_scells }; }
    Span<std::size_t>         ocells       () { return { reinterpret_cast<std::size_t *>( _fcells + nb_fcells + nb_scells ), nb_ocells }; }

    TI                        end_ind_in_fcells; ///< end index in final cells
    CellBounds                bounds;            ///< pos and weight bounds
    int                       nb_fcells;         ///< final cells
    int                       nb_scells;         ///< super cells
    int                       nb_ocells;         ///< out of core cells
    LGridFinalCell<Pc>*       _fcells[ 1 ];      ///<
};

// impl ------------------------------------------------------------------------------------
template<class Pc>
LGridSuperCell<Pc> *LGridSuperCell<Pc>::allocate( BumpPointerPool &pool, std::size_t &ram_acc, int nb_fcells, int nb_scells, int nb_ocells ) {
    std::size_t ram = sizeof( LGridSuperCell ) + ( nb_fcells - 1 + nb_scells + nb_ocells ) * sizeof( void * );
    LGridSuperCell *res = new ( pool.allocate( ram ) ) LGridSuperCell;
    res->nb_fcells = nb_fcells;
    res->nb_scells = nb_scells;
    res->nb_ocells = nb_ocells;
    ram_acc += ram;
    return res;
}

} // namespace sdot
