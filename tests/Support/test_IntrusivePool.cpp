#include "../src/sdot/Support/IntrusivePool.h"
#include "catch_main.h"

using namespace sdot;

TEST_CASE( "Pool" ) {
    struct Item { Item *prev_in_pool, *next_in_pool; std::size_t val; };
    IntrusivePool<Item,256> pool;

    // creation
    std::vector<Item *> items;
    for( std::size_t i = 0; i < 256; ++i ) {
        Item *item = pool.create();
        items.push_back( item );
        item->val = i;
    }

    // check memory
    for( std::size_t i = 0; i < 256; ++i )
        CHECK( items[ i ]->val == i );

    // traversal
    std::vector<int> count( 256, 0 );
    pool.foreach( [&]( Item &item ) {
        ++count[ item.val ];
    } );
    for( std::size_t i = 0; i < 256; ++i )
        CHECK( count[ i ] == 1 );

    // free
    for( std::size_t i = 0; i < 256; i += 2 )
        pool.free( items[ i ] );

    pool.foreach( [&]( Item &item ) {
        ++count[ item.val ];
    } );
    for( std::size_t i = 0; i < 256; ++i )
        CHECK( count[ i ] == 1 + ( i % 2 ) );
}

