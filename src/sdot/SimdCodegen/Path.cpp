#include "Path.h"

Path::Path( std::vector<Instruction *> instructions ) : instructions( instructions ) {
}

void Path::write_to_stream( std::ostream &os ) const {
    for( std::size_t i = 0; i < instructions.size(); ++i )
        instructions[ i ]->write_to_stream( os << "\n" );
}
