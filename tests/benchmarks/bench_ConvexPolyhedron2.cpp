//#include <vector>
//static std::vector<int> bc( 24, 0 );
#include "../../src/sdot/Geometry/ConvexPolyhedron2.h"
#include "../../src/sdot/Support/StaticRange.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
#include <fstream>
#include <map>
using namespace sdot;

// // nsmake cpp_flag -march=skylake
// // nsmake cxx_name clang++

//// nsmake cpp_flag -march=native
//// nsmake lib_flag -O3
//// nsmake cpp_flag -O3

static __attribute__ ((noinline))
std::uint64_t rdtsc_overhead_func( std::uint64_t dummy ) {
    return dummy;
}

template<class Cp,class Pt,int Simd,int Switch>
void bench( const std::vector<std::size_t> &offsets, const std::vector<std::pair<Pt,Pt>> &cuts, N<Simd>, N<Switch> ) {
    constexpr int flags = ConvexPolyhedron::do_not_use_simd     * ( Simd   == 0 ) +
                          ConvexPolyhedron::do_not_use_switches * ( Switch == 0 );
    using TF = typename Cp::TF;

    //
    std::uint64_t t0 = 0, t1 = 0;
    std::uint64_t rdtsc_overhead = (uint64_t)UINT64_MAX;
    for( std::size_t rep = 0; rep < 128; ++rep ) {
        RDTSC_START( t0 );
        rdtsc_overhead_func( 1 );
        RDTSC_FINAL( t1 );
        rdtsc_overhead = std::min( rdtsc_overhead, t1 - t0 );
    }

    // initial cell
    std::uint64_t dt = 0;
    Cp lc( typename Cp::Box{ { 0, 0 }, { 1, 1 } } ), cp;
    RDTSC_START( t0 );
    for( std::size_t num_point = 1; num_point < offsets.size(); ++num_point ) {
        cp = lc;
        for( std::size_t i = offsets[ num_point - 1 ]; i < offsets[ num_point ]; ++i ) {
            // bc[ cp.nb_nodes() ]++;
            cp.plane_cut( cuts[ i ].first, cuts[ i ].second, 17, N<flags>() );
        }
    }
    RDTSC_FINAL( t1 );
    dt += t1 - t0 - rdtsc_overhead;

    P( dt / double( cuts.size() ) );
    // P( bc );
}


int main() {
    struct Pc {
        enum { store_the_normals = false };
        enum { allow_ball_cut    = false };
        using  TF                = double;
        using  TI                = std::size_t;
        using  CI                = std::size_t;

    };
    using Cp = ConvexPolyhedron2<Pc>;
    using Pt = Cp::Pt;

    struct Dirac {
        bool operator<( const Dirac &that ) const {
            return std::tie( phase, pos.x, pos.y ) < std::tie( that.phase, that.pos.x, that.pos.y );
        };
        std::size_t phase;
        Pt          pos;
    };

    // read file
    std::ifstream fin( "tests/benchmarks/cuts.txt" );
    std::map<Dirac,std::vector<std::pair<Pt,Pt>>> pt_map;
    while ( true ) {
        Pt pos, o, n;
        std::size_t phase;
        fin >> phase >> pos.x >> pos.y
            >> o.x >> o.y
            >> n.x >> n.y;
        if ( ! fin )
            break;
        pt_map[ { phase, pos } ].emplace_back( o, n );
    }

    //
    std::vector<std::pair<Pt,Pt>> cuts;
    std::vector<std::size_t> offsets;
    for( const auto &p : pt_map ) {
        offsets.push_back( cuts.size() );
        for( auto v : p.second )
            cuts.push_back( v );
    }
    offsets.push_back( cuts.size() );


    bench<Cp>( offsets, cuts, /*simd*/ N<0>(), /*switch*/ N<0>() );
    bench<Cp>( offsets, cuts, /*simd*/ N<1>(), /*switch*/ N<0>() );
    bench<Cp>( offsets, cuts, /*simd*/ N<1>(), /*switch*/ N<1>() );
}
