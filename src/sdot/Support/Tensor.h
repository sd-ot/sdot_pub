#pragma once

#include <iostream>
#include <vector>
#include <array>

namespace sdot {

/** (very) simple tensor
 */
template<class T,int dim>
class Tensor {
public:
    using I              = std::size_t;
    using D              = std::vector<T>;
    using S              = std::array<I,dim>;

    /**/  Tensor         ( S size ) : size( size ) { I a = 1; for( I d = 0; d < dim; ++d ) { acc[ d ] = a; a *= size[ d ]; } data.resize( a ); }
    /**/  Tensor         () { for( I d = 0; d < dim; ++d ) size[ d ] = 0; }

    template<class P>
    T    &operator[]     ( const P &p ) { I o = 0; for( I d = 0; d < dim; ++d ) o += acc[ d ] * p[ d ]; return data[ o ]; }

    void  write_to_stream( std::ostream &os ) const { for( I i = 0; i < data.size(); ++i ) { if ( i ) disp_sep( os, i ); os << data[ i ]; } }
    void  disp_sep       ( std::ostream &os, I i ) const { I n = 0; for( I d = 1; d < dim; ++d ) n += ( i % acc[ d ] ) == 0; if ( n ) for( I d = 0; d < n; ++d ) os << "\n"; else os << " "; }

    D     data;          ///<
    S     size;          ///<
    S     acc;           ///<
};

} // namespace sdot
