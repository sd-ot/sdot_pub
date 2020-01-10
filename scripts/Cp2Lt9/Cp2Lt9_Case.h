#pragma once

#include "../../src/sdot/Support/OptParm.h"
#include "Cp2Lt9_CutList.h"
#include <string>

/***/
class Cp2Lt9_Case {
public:
    /**/              Cp2Lt9_Case  ( OptParm &opt_parm, int nb_nodes, unsigned comb, int simd_size, int nb_registers, bool pi_in_regs );
    /**/              Cp2Lt9_Case  ();

    void              make_code    ( std::ostream &os );

    std::string       val_reg      ( std::string c, int n );;

    // constraints
    int               nb_registers;
    bool              pi_in_regs;
    int               simd_size;
    std::vector<bool> outside;

    std::string       sp;

    // output
    Cp2Lt9_CutList    cut_list;
    bool              valid;
    std::string       code;
};

