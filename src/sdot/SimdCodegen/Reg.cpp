#include "Reg.h"

Reg::Reg( int size, std::string id ) : size( size ), id( id ) {
}

bool Reg::operator==( const Reg &that ) const {
    return id == that.id;
}

void Reg::write_to_stream( std::ostream &os ) const {
    os << id;
}
