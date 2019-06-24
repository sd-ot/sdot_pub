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
//// nsmake cpp_flag -O2

struct Pc {
    enum { store_the_normals = false };
    enum { allow_ball_cut    = false };
    using  TF                = double;
    using  TI                = std::size_t;
    using  CI                = std::size_t;

};

using Cp  = ConvexPolyhedron2<Pc>;
using Pt  = Cp::Pt;
using Cut = Cp::Cut;

struct Dirac {
    bool operator<( const Dirac &that ) const {
        return std::tie( phase, pos.x, pos.y ) < std::tie( that.phase, that.pos.x, that.pos.y );
    };
    std::size_t phase;
    Pt          pos;
};




static __attribute__ ((noinline))
double fake_cp_plane_cut( const Cut *cuts, std::size_t nb_cuts ) {
    double res = 0;
    for( std::size_t i = 0; i < nb_cuts; ++i )
        res += cuts[ i ].dir.x + cuts[ i ].dir.y + cuts[ i ].dist;
    return res;
}

template<int Simd,int Switch>
void bench( const std::vector<std::size_t> &offsets, const std::vector<Cut> &cuts, N<Simd>, N<Switch> ) {
    constexpr int flags = ConvexPolyhedron::do_not_use_simd     * ( Simd   == 0 ) +
                          ConvexPolyhedron::do_not_use_switch * ( Switch == 0 );

    // cells
    Cp lc( typename Cp::Box{ { 0, 0 }, { 1, 1 } } ), cp;

    // overhead
    double sum = 0;
    std::uint64_t t0 = 0, t1 = 0, nb_reps = 1280;
    RDTSC_START( t0 );
    for( std::size_t rep = 0; rep < nb_reps; ++rep ) {
        for( std::size_t num_point = 1; num_point < offsets.size(); ++num_point ) {
            cp = lc;
            sum += fake_cp_plane_cut( cuts.data() + offsets[ num_point - 1 ], offsets[ num_point ] - offsets[ num_point - 1 ] );
        }
    }
    RDTSC_FINAL( t1 );
    std::uint64_t overhead = ( t1 - t0 ) / nb_reps;

    // cuts
    RDTSC_START( t0 );
    for( std::size_t rep = 0; rep < nb_reps; ++rep ) {
        for( std::size_t num_point = 1; num_point < offsets.size(); ++num_point ) {
            cp = lc;
            cp.plane_cut( cuts.data() + offsets[ num_point - 1 ], offsets[ num_point ] - offsets[ num_point - 1 ], N<flags>() );
            sum += cp.nb_nodes();
        }
    }
    RDTSC_FINAL( t1 );
    std::uint64_t dt = ( t1 - t0 ) / nb_reps - overhead;

    P( sum, dt, overhead, dt / double( cuts.size() ) );
    // P( bc );
}


int main() {
    // read file
    std::ifstream fin( "tests/benchmarks/cuts.txt" );
    std::map<Dirac,std::vector<Cut>> pt_map;
    while ( true ) {
        Pt pos, o, n;
        std::size_t phase;
        fin >> phase >> pos.x >> pos.y
            >> o.x >> o.y
            >> n.x >> n.y;
        if ( ! fin )
            break;
        pt_map[ { phase, pos } ].push_back( { n, dot( o, n ), 17 } );
    }

    //
    std::vector<Cut> cuts;
    std::vector<std::size_t> offsets;
    for( const auto &p : pt_map ) {
        offsets.push_back( cuts.size() );
        for( auto v : p.second )
            cuts.push_back( v );
    }
    offsets.push_back( cuts.size() );

    //    bench( offsets, cuts, /*simd*/ N<0>(), /*switch*/ N<0>() );
    //    bench( offsets, cuts, /*simd*/ N<1>(), /*switch*/ N<0>() );
    bench( offsets, cuts, /*simd*/ N<1>(), /*switch*/ N<1>() );
}
