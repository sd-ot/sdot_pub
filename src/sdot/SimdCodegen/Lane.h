#pragma once

#include "Reg.h"

/**
*/
class Lane {
public:
    /**/ Lane           ( Reg reg, int lane );

    void write_to_stream( std::ostream &os ) const;

    Reg  reg;
    int  lane;
};

