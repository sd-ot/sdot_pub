#pragma once

#include "../src/sdot/Support/SimdVec.h"
#include "../src/sdot/Support/Time.h"
#include "../src/sdot/Support/P.h"
#include <iostream>
#include <fstream>
#include <vector>

using TFunc = double( int *cases, int nb_cases );

double _timing( TFunc *func, int *cases_data, int cases_size, std::uint64_t nb_reps = 50 ) {
    std::uint64_t res = -1ul;
    for( std::uint64_t rep = 0, t0 = 0, t1 = 0; rep < nb_reps; ++rep ) {
        RDTSC_START( t0 );
        double r = func( cases_data, cases_size );
        RDTSC_FINAL( t1 );
        res = std::min( res, t1 - t0 );
        if ( r == 5425414214.8 )
            P( r );
    }
    return res;
};

void bench_Cp2Lt64_code( TFunc *func, int nb_cases, const char *output_filename ) {
    std::vector<double> timings( nb_cases, 1e40 );
    std::vector<int> cases( 100 );
    double overhead = 1e40;

    for( std::size_t rep = 0; rep < 1000; ++rep ) {
        overhead = std::min( overhead, _timing( func, cases.data(), 0 ) );

        for( int num_case = 0; num_case < nb_cases; ++num_case ) {
            for( std::size_t i = 0; i < cases.size(); ++i )
                cases[ i ] = num_case + 2;
            timings[ num_case ] = std::min( timings[ num_case ], _timing( func, cases.data(), cases.size() ) );
        }
    }

    for( std::size_t i = 0; i < timings.size(); ++i )
        timings[ i ] = ( timings[ i ] - overhead ) / cases.size();
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
