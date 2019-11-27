#pragma once

#include "LGridParentCell.h"
#include <unistd.h>

namespace sdot {

/**
*/
template<class Pc>
struct LGridOutOfCoreCell : LGridParentCell<Pc> {
    /**/                      ~LGridOutOfCoreCell() { if ( ! filename.empty() ) unlink( filename.c_str() ); }
    static LGridOutOfCoreCell *allocate          ( LGridBaseCell<Pc> *sub_cell );
    void                       clear             () { delete this->sub_cells[ 0 ]; this->sub_cells[ 0 ] = nullptr; }

    std::string                filename;         ///<
    bool                       modified;         ///<
    char                      *alloc;            ///<
    LGridOutOfCoreCell        *prev;             ///<
};

template<class Pc>
LGridOutOfCoreCell<Pc> *LGridOutOfCoreCell<Pc>::allocate( LGridBaseCell<Pc> *sub_cell ) {
    LGridOutOfCoreCell<Pc> *res = new LGridOutOfCoreCell<Pc>;
    res->sub_cells[ 0 ] = sub_cell;
    res->nb_sub_items = 0;
    res->modified = true;

    return res;
}

} // namespace sdot
