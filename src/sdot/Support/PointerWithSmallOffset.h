#pragma once

#include <cstdint>

namespace sdot {

/**
*/
template<class T,int m>
class PointerWithSmallOffset {
public:
    union       U                     { T *p; std::size_t o; };

    /**/        PointerWithSmallOffset( T *p, std::size_t o = 0 ) { u.p = p; u.o += o; }
    /**/        PointerWithSmallOffset() {}

    std::size_t offset                () const { return u.o & ( m - 1 ); }
    T          *ptr                   () const { U res = u; res.o &= ~( m - 1 ); return res.p; }

    operator    bool                  () const { return u.p; }

private:
    U           u;
};

} // namespace sdot
