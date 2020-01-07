#pragma once

#include "../../src/sdot/Support/OptParm.h"
#include <string>

/**
*/
class Cp2Lt9_Func {
public:
    /***/       Cp2Lt9_Func( OptParm &opt_parm, std::string float_type, std::string simd_type, int max_nb_nodes = 9 );

    std::string float_type;
    std::string simd_type;

    int         simd_size;  ///< size of the registers
    bool        make_di;    ///< true to make di_xx direclty (instead of bi_xx > cs, and di = bi_xx - cs later)

    double      score;      ///< weight time
};

