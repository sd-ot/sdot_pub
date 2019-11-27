#pragma once

#include "../../Support/BumpPointerPool.h"
#include "LGridBaseCell.h"
#include <unistd.h>

namespace sdot {

/**
*/
template<class Pc>
struct LGridOutOfCoreCell : LGridBaseCell<Pc> {
    /**/              ~LGridOutOfCoreCell() { if ( ! filename.empty() ) unlink( filename.c_str() ); }

    BumpPointerPool    mem_pool_cells;   ///< to store the sub cells
    LGridBaseCell<Pc> *sub_cell;         ///< if available (or null)
    std::string        filename;         ///<
};

} // namespace sdot
