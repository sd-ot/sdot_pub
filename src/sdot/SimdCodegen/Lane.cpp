#include "../Support/Display/generic_ostream_output.h"
#include "Lane.h"

Lane::Lane( Reg reg, int lane ) : reg( reg ), lane( lane ) {
}

void Lane::write_to_stream( std::ostream &os ) const {
    os << reg << "[" << lane << "]";
}
