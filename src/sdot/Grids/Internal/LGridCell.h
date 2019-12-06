#pragma once

#include "../../Support/BumpPointerPool.h"
#include "../../Support/Span.h"
#include "CellBoundsTraits.h"

namespace sdot {

/**
*/
template<class Pc>
struct LGridCell {
    using                     CellBounds      = typename CellBoundsTraits<Pc>::type;
    using                     Dirac           = typename Pc::Dirac;
    using                     TI              = typename Pc::TI;
    struct                    FirstAllocData  { LGridCell *next, *parent; std::size_t num_in_parent; };
    union                     Child           { LGridCell *ptr; std::size_t off; };
    union                     Data            { Child children[ 0 ]; Dirac diracs[ 0 ]; };

    /**/                      LGridCell       () : data{} {}

    std::size_t               size_in_bytes   ( bool first_alloc ) const;
    static LGridCell*         allocate        ( BumpPointerPool &pool, std::size_t &ram, int nb_diracs, int nb_scells, int nb_ocells ); ///<

    FirstAllocData&           first_alloc_data() { return *reinterpret_cast<FirstAllocData *>( reinterpret_cast<char *>( this ) + size_in_bytes( false ) ); }
    Dirac&                    dirac           ( int index ) { return data.diracs[ index ]; }
    LGridCell*&               scell           ( int index ) { return data.children[ index ].ptr; }
    std::size_t&              ocell           ( int index ) { return data.children[ nb_scells + index ].off; }

    Span<Dirac>               diracs          () { return { data.diracs, std::size_t( nb_diracs ) }; }
    Span<LGridCell*>          scells          () { return { &data.children[ 0 ].ptr, std::size_t( nb_scells ) }; }
    Span<std::size_t>         ocells          () { return { &data.children[ 0 ].off + nb_scells , std::size_t( nb_ocells ) }; }

    TI                        end_ind_in_fcells; ///< end index in final cells
    int                       nb_diracs;         ///< diracs values
    int                       nb_scells;         ///< super cells ( data.children[ num ].ptr )
    int                       nb_ocells;         ///< out of core cells ( data.children[ nb_scells + num ].off )
    CellBounds                bounds;            ///< for pos and weight
    Data                      data;              ///<
};

// impl ------------------------------------------------------------------------------------
template<class Pc>
LGridCell<Pc> *LGridCell<Pc>::allocate( BumpPointerPool &pool, std::size_t &ram_acc, int nb_diracs, int nb_scells, int nb_ocells ) {
    std::size_t ram = sizeof( LGridCell ) - sizeof( Data ) + nb_scells * sizeof( LGridCell * ) + nb_ocells * sizeof( std::size_t ) + nb_diracs * sizeof( Dirac ) + sizeof( FirstAllocData );
    LGridCell *res = new ( pool.allocate( ram ) ) LGridCell;
    res->nb_diracs = nb_diracs;
    res->nb_scells = nb_scells;
    res->nb_ocells = nb_ocells;
    ram_acc += ram;

    res->first_alloc_data().parent = nullptr;
    res->first_alloc_data().next = nullptr;

    return res;
}

template<class Pc>
std::size_t LGridCell<Pc>::size_in_bytes( bool first_alloc ) const {
    return sizeof( LGridCell ) - sizeof( Data ) +
           nb_scells * sizeof( LGridCell * ) +
           nb_ocells * sizeof( std::size_t ) +
           nb_diracs * sizeof( Dirac ) +
           first_alloc * sizeof( FirstAllocData );
}

} // namespace sdot
