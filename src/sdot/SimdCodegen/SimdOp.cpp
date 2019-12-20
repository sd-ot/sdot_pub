#include "SimdOp.h"

SimdOp::SimdOp( std::string name, const std::vector<SimdOp *> &children ) : children( children ), name( name ) {
    op_id = 0;
}

void SimdOp::write_to_stream( std::ostream &os ) const {
    os << name;
}
