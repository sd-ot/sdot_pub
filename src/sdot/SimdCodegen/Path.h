#pragma once

#include "Instruction.h"
#include <vector>

class Path {
public:
    /**/                       Path           ( std::vector<Instruction *> instructions = {} );

    void                       write_to_stream( std::ostream &os ) const;

    std::vector<Instruction *> instructions; ///<
};

