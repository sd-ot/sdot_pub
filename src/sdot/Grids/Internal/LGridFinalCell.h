#pragma  once

#include "LGridBaseCell.h"

namespace sdot {

/**
*/
template<class Pc>
struct LGridFinalCell : LGridBaseCell<Pc> {
    using  BaseCell     = LGridBaseCell<Pc>;
    using  Dirac        = typename Pc::Dirac;

    int    nb_diracs    () const { return this->nb_sub_items; }

    Dirac *diracs[ 1 ]; ///<
};

} // namespace sdot
