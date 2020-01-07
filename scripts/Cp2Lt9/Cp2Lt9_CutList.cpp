#include "Cp2Lt9_CutList.h"


Cp2Lt9_CutList::Cp2Lt9_CutList( std::vector<bool> outside ) {
    for( std::size_t i = 0; i < outside.size(); ++i ) {
        if ( outside[ i ] )
            continue;

        // going inside
        std::size_t h = ( i + outside.size() - 1 ) % outside.size();
        if ( outside[ h ] )
            ops.push_back( { h, i, 1 } );

        // inside point
        ops.push_back( { i, i, 0 } );

        // outside point => create points on boundaries
        std::size_t j = ( i + 1 ) % outside.size();
        if ( outside[ j ] )
            ops.push_back( { i, j, 1 } );
    }
}

void Cp2Lt9_CutList::write_to_stream( std::ostream &os ) const {
    os << ops;
}

std::vector<std::size_t> Cp2Lt9_CutList::split_indices() const {
    std::vector<std::size_t> res;
    for( std::size_t i = 0; i < ops.size(); ++i )
        if ( ops[ i ].split() )
            res.push_back( i );
    return res;
}

void Cp2Lt9_CutList::rotate( std::size_t off ) {
    std::vector<Cut> nops( ops.size() );
    for( std::size_t i = 0; i < ops.size(); ++i )
        nops[ i ] = ops[ ( i + off ) % ops.size() ];
    ops = nops;
}

void Cp2Lt9_CutList::sw( uint64_t val ) {
    std::vector<std::size_t> si = split_indices();
    for( std::size_t i = 0; i < si.size(); ++i )
        ops[ si[ i ] ].sw = val & ( std::uint64_t( 1 ) << i );
}
