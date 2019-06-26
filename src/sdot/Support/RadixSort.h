#pragma once

#include "StaticRange.h"
#include "ThreadPool.h"
#include "ASSERT.h"
#include <immintrin.h>
#include <cstring>
#include <array>

namespace sdot {

//#ifdef __AVX512F__
///**
// * avx512 radix_sort
//*/
//#endif

/**
 * generic radix_sort
*/
template<class TK,class TV,int nb_bits>
std::pair<TK *,TV *> radix_sort( std::pair<TK *,TV *> out, std::pair<TK *,TV *> inp, std::size_t len, N<nb_bits>, std::vector<std::size_t> &tmps ) {
    constexpr std::size_t nb_bits_per_shift = 12, nb_buckets = 1 << nb_bits_per_shift, nb_rounds = ( nb_bits + nb_bits_per_shift - 1 ) / nb_bits_per_shift;
    std::size_t nb_threads = thread_pool.nb_threads(), nb_jobs = nb_threads;
    tmps.resize( nb_buckets * nb_jobs );

    StaticRange<nb_rounds>::for_each( [&]( auto n ) {
        constexpr int shift = nb_bits_per_shift * n.val;
        
        // get count for each bucket
        for( std::size_t i = 0; i < tmps.size(); ++i )
            tmps[ i ] = 0;
        thread_pool.execute( nb_jobs, [&]( std::size_t num_job, int ) {
            std::size_t *tmp = tmps.data() + nb_buckets * num_job;
            std::size_t beg = ( num_job + 0 ) * len / nb_jobs;
            std::size_t end = ( num_job + 1 ) * len / nb_jobs;
            for( std::size_t i = beg; i < end; ++i ) {
                auto s = inp.first[ i ] >> shift;
                if ( n != nb_rounds - 1 )
                    s &= nb_buckets - 1;
                ++tmp[ s ];
            }
        } );

        // suffix sum
        for( std::size_t i = 0, acc = 0; i < nb_buckets; ++i ) {
            for( std::size_t num_job = 0; num_job < nb_jobs; ++num_job ) {
                std::size_t *tmp = tmps.data() + nb_buckets * num_job;
                std::size_t v = acc;
                acc += tmp[ i ];
                tmp[ i ] = v;
            }
        }

        // save
        thread_pool.execute( nb_jobs, [&]( std::size_t num_job, int ) {
            std::size_t *tmp = tmps.data() + nb_buckets * num_job;
            std::size_t beg = ( num_job + 0 ) * len / nb_jobs;
            std::size_t end = ( num_job + 1 ) * len / nb_jobs;
            for( std::size_t i = beg; i < end; ++i ) {
                auto s = inp.first[ i ] >> shift;
                if ( n != nb_rounds - 1 )
                    s &= nb_buckets - 1;
                std::size_t index = tmp[ s ]++;
                out.first[ index ] = inp.first[ i ];
                out.second[ index ] = inp.second[ i ];
            }
        } );
        
        std::swap( out, inp );
    } );

    return inp;
}

template<class TK,class TV,int nb_bytes>
std::pair<TK *,TV *> radix_sort( std::pair<TK *,TV *> out, std::pair<TK *,TV *> inp, std::size_t len, N<nb_bytes> n ) {
    std::vector<std::size_t> tmps;
    return radix_sort( out, inp, len, n, tmps );
}

}
