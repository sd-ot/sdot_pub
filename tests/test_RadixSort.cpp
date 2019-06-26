#include "../src/sdot/Support/RadixSort.h"
#include "catch_main.h"

using namespace sdot;
//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O3

//template<class TI>
//void test_with_type() {
//    std::size_t n = 100000000ul;
//    std::vector<TI> inp;
//    for( std::size_t i = 0; i < n; ++i )
//        inp.push_back( rand() );

//    std::vector<TI> out( inp.size() );
//    tick.start( "radix" );
//    N<8*sizeof(TI)> ns;
//    TI *res = radix_sort( out.data(), inp.data(), inp.size(), ns );
//    tick.stop( "radix" );
//    for( std::size_t i = 1; i < inp.size(); ++i )
//        EXPECT_GE( res[ i ], res[ i - 1 ] );
//    //    P( std::vector<TI>{ res, res + inp.size() } );

//    inp.resize( 0 );
//    for( std::size_t i = 0; i < n; ++i )
//        inp.push_back( rand() );

//    tick.start( "sort" );
//    std::sort( inp.begin(), inp.end() );
//    tick.stop( "sort" );
//}

//TEST( RadixSort, dim_eq_2 ) {
//    //    test_with_type<std::uint8_t >();
//    //    test_with_type<std::uint16_t>();
//    test_with_type<std::uint32_t>();
//}

TEST_CASE( "radix" ) {
    using TI = std::uint64_t;
    std::size_t n = 100ul;
    std::vector<TI> inp_keys;
    std::vector<TI> inp_vals;
    for( std::size_t i = 0; i < n; ++i ) {
        TI v = rand() % 100;
        inp_keys.push_back( v );
        inp_vals.push_back( v + 100 );
    }

    std::vector<TI> out_keys( inp_keys.size() );
    std::vector<TI> out_vals( inp_vals.size() );
    std::pair<TI *,TI *> res = radix_sort( std::make_pair( out_keys.data(), out_vals.data() ), std::make_pair( inp_keys.data(), inp_vals.data() ), inp_keys.size(), N<8*sizeof(TI)>() );

    for( std::size_t i = 0; i < inp_keys.size(); ++i ) {
        CHECK( res.first [ i ] <  100 );
        CHECK( res.second[ i ] >= 100 );
        CHECK( res.second[ i ] <  200 );
    }

    for( std::size_t i = 1; i < inp_keys.size(); ++i ) {
        CHECK( res.first [ i ] >= res.first [ i - 1 ] );
        CHECK( res.second[ i ] >= res.second[ i - 1 ] );
    }
}

