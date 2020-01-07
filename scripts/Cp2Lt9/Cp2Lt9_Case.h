#pragma once

#include "../../src/sdot/Support/OptParm.h"
#include "Cp2Lt9_CutList.h"
#include <string>

/***/
class Cp2Lt9_Case {
public:
    /**/              Cp2Lt9_Case  ( OptParm &opt_parm, int nb_nodes, unsigned comb, int simd_size );
    /**/              Cp2Lt9_Case  ();

    void              make_code    ( std::ostream &os );

    // constraints
    int               simd_size;
    std::vector<bool> outside;

    std::string       sp;

    // output
    Cp2Lt9_CutList    cut_list;
    bool              valid;
    std::string       code;
};

