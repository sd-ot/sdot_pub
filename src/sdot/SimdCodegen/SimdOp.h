#pragma once

#include <vector>
#include <string>

/**
  GET n
  REG name type nb_lanes
  ADD
  SUB
  MUL
  DIV
*/
class SimdOp {
public:
    /* */                 SimdOp         ( std::string name, const std::vector<SimdOp *> &children, std::uint64_t op_id = 0 );

    void                  write_to_stream( std::ostream &os ) const;
    void                  write_code     ( std::ostream &os, std::string sp, int &nb_regs );
    std::ostream         &write_reg      ( std::ostream &os );

    bool                  better_code    ( SimdOp *that );
    bool                  bin_op         () const;
    bool                  set_op         () const;

    std::vector<SimdOp *> children;
    std::uint64_t         op_id;
    std::string           name;

    std::vector<SimdOp *> parents;
    int                   nreg;
    mutable SimdOp       *repl;
};

