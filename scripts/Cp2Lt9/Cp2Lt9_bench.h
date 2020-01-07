#pragma once

#include "../src/sdot/Support/SimdVec.h"
#include "../src/sdot/Support/Time.h"
#include "../src/sdot/Support/P.h"
#include <functional>
#include <iostream>
#include <fstream>
#include <vector>

template<class TF,class TC>
double _timing( std::function<void( TF *px, TF *py, TC *pi, int &nodes_size, const TF *cut_x, const TF *cut_y, const TF *cut_s, const TC *cut_i, int cut_n )> func, TF *px, TF *py, TC *pi, int nodes_size, TF *cut_x, TF *cut_y, TF *cut_s, TC *cut_i, int cut_n, std::uint64_t nb_reps = 50 ) {
    std::uint64_t res = -1ul;
    for( std::uint64_t rep = 0, t0 = 0, t1 = 0; rep < nb_reps; ++rep ) {
        RDTSC_START( t0 );
        func( px, py, pi, nodes_size, cut_x, cut_y, cut_s, cut_i, cut_n );
        RDTSC_FINAL( t1 );
        res = std::min( res, t1 - t0 );
    }
    return res;
};

template<typename TF,typename TC>
void bench_Cp2Lt64_code( std::vector<std::function<void( TF *px, TF *py, TC *pi, int &nodes_size, const TF *cut_x, const TF *cut_y, const TF *cut_s, const TC *cut_i, int cut_n )>> funcs, int nodes_size, const char *output_filename ) {
    alignas( 64 ) TF px[ 64 ];
    alignas( 64 ) TF py[ 64 ];
    alignas( 64 ) TC pi[ 64 ];
    alignas( 64 ) TF cut_x[ 64 ];
    alignas( 64 ) TF cut_y[ 64 ];
    alignas( 64 ) TF cut_s[ 64 ];
    alignas( 64 ) TC cut_i[ 64 ];
    int cut_n = 64;

    for( int i = 0; i < cut_n; ++i ) {
        px[ i ] = 0;
        py[ i ] = 0;
        pi[ i ] = 0;
        cut_x[ i ] = 1;
        cut_y[ i ] = 0;
        cut_s[ i ] = 2;
        cut_i[ i ] = 0;
    }

    // the "fully outside" case is used to reinitialize px. Reinitialization is counted as overhead
    //    double overhead = 1e40;
    //    for( int i = 0; i < cut_n; ++i )
    //    for( std::size_t rep = 0; rep < 1000; ++rep )
    //        overhead = std::min( overhead, _timing( funcs[ 0 ], px, py, pi, nodes_size, cut_x, cut_y, cut_s, cut_i, cut_n / 2 ) );
    double overhead = 0;

    // get timings (with reinitialization before each step)
    std::vector<double> timings( funcs.size(), 1e40 );
    for( int i = 1; i < cut_n; i += 2 )
        cut_s[ i ] = 0;
    for( std::size_t rep = 0; rep < 15000; ++rep )
        for( std::size_t num_func = 0; num_func < funcs.size(); ++num_func )
            timings[ num_func ] = std::min( timings[ num_func ], _timing( funcs[ num_func ], px, py, pi, nodes_size, cut_x, cut_y, cut_s, cut_i, cut_n ) );

    // timings without overhead
    for( std::size_t num_func = 0; num_func < timings.size(); ++num_func )
        timings[ num_func ] = ( timings[ num_func ] - overhead ) / cut_n;
    P( timings );

    std::size_t best_i = 0;
    for( std::size_t i = 1; i < timings.size(); ++i )
        if ( timings[ best_i ] > timings[ i ] )
            best_i = i;
    P( timings[ best_i ] );
    P( best_i );

    if ( output_filename ) {
        std::ofstream fout( output_filename );
        fout << best_i << " " << timings[ best_i ] << "\n";
    }
}
