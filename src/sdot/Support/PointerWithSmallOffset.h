#pragma once

#include <cstdint>

namespace sdot {

/**
*/
template<class T,int m>
class PointerWithSmallOffset {
public:
    union                  U                     { T *p; std::size_t o; };

    /**/                   PointerWithSmallOffset( const PointerWithSmallOffset &p ) : u( p.u ) {}
    /**/                   PointerWithSmallOffset( T *p, std::size_t o = 0 ) { u.p = p; u.o += o; }
    /**/                   PointerWithSmallOffset() {}

    PointerWithSmallOffset with_1_xored_offset   () const { PointerWithSmallOffset res; res.u.o = u.o ^ 1; return res; }
    std::size_t            offset                () const { return u.o & ( m - 1 ); }
    operator               bool                  () const { return u.p; }
    T                     *ptr                   () const { U res = u; res.o &= ~( m - 1 ); return res.p; }


private:
    U                      u;
};

} // namespace sdot
