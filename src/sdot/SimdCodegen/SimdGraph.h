#pragma once

#include <functional>
#include "SimdOp.h"
#include <deque>

/**
*/
class SimdGraph {
public:
    /**/                  SimdGraph         ();

    void                  for_each_child    ( const std::function<void(SimdOp *)> &f, const std::vector<SimdOp *> &targets );

    void                  add_target        ( SimdOp *target );
    SimdOp               *make_op           ( std::string name, const std::vector<SimdOp *> &children );

    void                  display           ( std::string filename = ".tmp" );

private:
    void                  for_each_child_rec( const std::function<void(SimdOp *)> &f, SimdOp *target );

    std::uint64_t         cur_op_id;
    std::vector<SimdOp *> targets;
    std::deque<SimdOp>    pool;
};

