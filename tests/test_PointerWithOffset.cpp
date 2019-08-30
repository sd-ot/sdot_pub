#include "../src/sdot/Support/PointerWithSmallOffset.h"
#include "catch_main.h"

using namespace sdot;

TEST_CASE( "PointerWithOffset" ) {
    PointerWithOffset<int> p( new int, 3 );

    CHECK( reinterpret_cast<std::size_t>( p.ptr() ) % 4 == 0 );
    CHECK( p.offset() == 3 );

    *p.ptr() = 5;
    CHECK( *p.ptr() == 5 );
}

