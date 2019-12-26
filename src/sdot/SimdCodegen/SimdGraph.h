#pragma once

#include <functional>
#include "SimdOp.h"
#include <deque>

/**
*/
class SimdGraph {
public:
    /**/                  SimdGraph         ( const SimdGraph &that );
    /**/                  SimdGraph         ();

    void                  for_each_child    ( const std::function<void(SimdOp *)> &f, const std::vector<SimdOp *> &targets, bool postfix = false ) const;

    void                  add_target        ( SimdOp *target );
    void                  write_code        ( std::ostream &os, std::string sp );
    SimdOp               *make_op           ( std::string name, const std::vector<SimdOp *> &children );
    SimdOp               *get_op            ( SimdOp *op, int num );

    void                  display           ( std::string filename = ".tmp" );

private:
    void                  for_each_child_rec( const std::function<void(SimdOp *)> &f, SimdOp *target, bool postfix ) const;
    void                  update_parents    ( std::vector<SimdOp *> *front = nullptr );

    mutable std::uint64_t cur_op_id;
    std::vector<SimdOp *> targets;
    std::deque<SimdOp>    pool;
};

