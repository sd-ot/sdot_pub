#include "../../src/sdot/Support/Display/binary_repr.h"
#include "../../src/sdot/Support/BumpPointerPool.h"
#include "../../src/sdot/Support/SimdRange.h"
#include "../../src/sdot/Support/SimdVec.h"
#include "../../src/sdot/Support/P.h"
#include "../catch_main.h"

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O3

using namespace sdot;

TEST_CASE( "simd addition" ) {
    int l = 47;
    BumpPointerPool pool;
    double *p = reinterpret_cast<double *>( pool.allocate( sizeof( double ) * l, sizeof( double ) * 8 ) );
    for( int i = 0; i < l; ++i )
        p[ i ] = i;

    SimdRange<8>::for_each( l, [&]( int i, auto s ) {
        using LF = SimdVec<double,s.val>;
        LF::store_aligned( p + i, LF::load_aligned( p + i ) * LF( 2 ) );
    } );

    for( int i = 0; i < l; ++i )
        CHECK( p[ i ] == 2 * i );
}

TEST_CASE( "simd cmp" ) {
    int l = 47;
    BumpPointerPool pool;
    double *p = reinterpret_cast<double *>( pool.allocate( sizeof( double ) * l, sizeof( double ) * 8 ) );
    for( int i = 0; i < l; ++i )
        p[ i ] = 2 * ( i % 2 );

    SimdRange<8>::for_each( l, [&]( int i, auto s ) {
        using LF = SimdVec<double,s.val>;
        std::uint64_t c = LF::load_aligned( p + i ) > LF( 1 );
        for( int j = 0; j < s.val; ++j )
            CHECK( ( ( c >> j ) & 1 ) == j % 2 );
    } );
}

