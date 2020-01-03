#include "../Support/tokenize.h"
#include "../Support/TODO.h"
#include "../Support/P.h"
#include "SimdOp.h"

SimdOp::SimdOp( std::string name, const std::vector<SimdOp *> &children, std::uint64_t op_id ) : children( children ), op_id( op_id ), name( name ) {
    nreg = -1;
}

void SimdOp::write_to_stream( std::ostream &os ) const {
    os << name;
}

void SimdOp::write_code( std::ostream &os, std::string sp, int &nb_regs ) {
    if ( bin_op() ) {
        nreg = nb_regs++;
        os << sp << "auto R" << nreg << " = ";
        children[ 0 ]->write_reg( os );
        if ( name == "ADD" ) os << " + ";
        if ( name == "SUB" ) os << " - ";
        if ( name == "MUL" ) os << " * ";
        if ( name == "DIV" ) os << " / ";
        children[ 1 ]->write_reg( os );
        os << ";\n";
        return;
    }

    auto vs = tokenize( name );
    if ( vs[ 0 ] == "GET" ) {
        nreg = nb_regs++;
        os << sp << "auto R" << nreg << " = ";
        children[ 0 ]->write_reg( os );
        os << "[ " << vs[ 1 ] << " ];\n";
        return;
    }

    if ( vs[ 0 ] == "SET" ) {
        os << sp << vs[ 1 ] << " = ";
        children[ 0 ]->write_reg( os );
        os << ";\n";
        return;
    }

    if ( vs[ 0 ] == "AGG" ) {
        nreg = nb_regs++;
        os << sp << "SimdVec<" << ( vs.size() >= 2 ? vs[ 1 ] : "TF" ) << "," << children.size() << "> R" << nreg << "{";
        for( std::size_t i = 0; i < children.size(); ++i )
            children[ i ]->write_reg( os << ( i ? ", " : " " ) );
        os << " };\n";
        return;
    }

    if ( vs[ 0 ] == "REG" )
        return;

    if ( vs[ 0 ] == "UNK" )
        return;

    P( name );
    TODO;
}


std::ostream &SimdOp::write_reg( std::ostream &os ) {
    if ( nreg >= 0 )
        return os << "R" << nreg;

    auto vs = tokenize( name );
    if ( vs[ 0 ] == "REG" )
        return os << vs[ 1 ];

    return os << "FEZR";
}

bool SimdOp::better_code( SimdOp *that ) {
    return std::make_tuple( this->set_op() ) < std::make_tuple( that->set_op() );
}

bool SimdOp::bin_op() const {
    return name == "ADD" || name == "SUB" || name == "MUL" || name == "DIV";
}

bool SimdOp::set_op() const {
    auto vs = tokenize( name );
    return vs[ 0 ] == "SET";
}
