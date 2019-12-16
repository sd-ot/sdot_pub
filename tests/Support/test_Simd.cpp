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

TEST_CASE( "simd aggregate double" ) {
    using SV = SimdVec<double,16>;
    SV s = SV::iota();

    int cpt = 0;
    for( auto v : s )
        CHECK( v == cpt++ );

    CHECK( ( s >  7 ) == 65280 );
    CHECK( ( s > 12 ) == 57344 );
}


TEST_CASE( "simd aggregate std::uint64_t 16" ) {
    using SV = SimdVec<std::uint64_t,16>;
    SV s = SV::iota();

    std::uint64_t cpt = 0;
    for( auto v : s )
        CHECK( v == cpt++ );

    cpt = 0;
    for( auto v : ( s << SV( 1 ) ) )
        CHECK( v == 2 * cpt++ );
}



TEMPLATE_TEST_CASE( "simd ctor several vars", "", std::uint32_t, std::uint64_t, float, double, long double ) {
    SimdVec<TestType,2> a( 1, 2 );
    CHECK( a[ 0 ] == 1 );
    CHECK( a[ 1 ] == 2 );

    alignas( 16 ) TestType v[] = { 10, 20 };
    SimdVec<TestType,2> b = SimdVec<TestType,2>::load_aligned( v );
    CHECK( b[ 0 ] == 10 );
    CHECK( b[ 1 ] == 20 );
}

