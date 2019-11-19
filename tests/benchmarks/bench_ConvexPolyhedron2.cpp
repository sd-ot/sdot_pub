#include "../../src/sdot/Geometry/ConvexPolyhedron2.h"
#include "../../src/sdot/Support/StaticRange.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
#include <fstream>
#include <map>
using namespace sdot;

// // nsmake cxx_name clang++
// // nsmake cpp_flag -march=skylake

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O3

struct Pc {
    enum { store_the_normals = false };
    enum { allow_ball_cut    = false };
    using  TF                = double;
    using  TI                = std::size_t;
    using  CI                = std::size_t;
    using  Pt                = Point2<TF>;

    struct Dirac {
        TF weight;
        Pt pos;
    };
};

using Cp = ConvexPolyhedron2<Pc>;
using TF = Cp::TF;
using Pt = Cp::Pt;

//static __attribute__ ((noinline))
//double fake_cp_plane_cut( const Cut *cuts, std::size_t nb_cuts ) {
//    double res = 0;
//    for( std::size_t i = 0; i < nb_cuts; ++i )
//        res += cuts[ i ].dir.x + cuts[ i ].dir.y + cuts[ i ].dist;
//    return res;
//}

template<int Simd,int Switch>
void bench( std::vector<TF> xs, std::vector<TF> ys, std::vector<TF> ps, std::vector<Pc::Dirac *> ds, N<Simd>, N<Switch> ) {
    constexpr int flags = ConvexPolyhedron::do_not_use_simd   * ( Simd   == 0 ) +
                          ConvexPolyhedron::do_not_use_switch * ( Switch == 0 );

    // overhead
    Cp cp;
    TF sum = 0;
    Cp::Box box{ { -1, -1 }, { 1, 1 } };
    std::uint64_t t0 = 0, t1 = 0, nb_reps = 1280000;
    RDTSC_START( t0 );
    for( std::size_t rep = 0; rep < nb_reps; ++rep ) {
        cp = box;
    }
    RDTSC_FINAL( t1 );
    std::uint64_t overhead = t1 - t0;

    // cuts
    RDTSC_START( t0 );
    for( std::size_t rep = 0; rep < nb_reps; ++rep ) {
        cp = box;
        cp.plane_cut( { xs.data(), ys.data() }, ps.data(), ds.data(), xs.size(), N<flags>() );
        sum += cp.nb_nodes();
    }
    RDTSC_FINAL( t1 );
    std::uint64_t dt = ( t1 - t0 - overhead ) / ( nb_reps * xs.size() );

    P( sum, overhead, t1 - t0, dt );
    P( cp );
}


int main() {
    std::size_t nb_cuts = 120;
    std::vector<TF> xs, ys, ps;
    std::vector<Pc::Dirac *> ds;
    for( std::size_t n = 0; n < nb_cuts; ++n ) {
        TF th = ( random() % 8 ) * M_PI / 5;
        xs.push_back( cos( th ) );
        ys.push_back( sin( th ) );
        ps.push_back( 10.0 / ( 11 + n ) );
        ds.push_back( nullptr );
    }

    bench( xs, ys, ps, ds, /*simd*/ N<0>(), /*switch*/ N<0>() );
//    bench( xs, ys, ps, ds, /*simd*/ N<1>(), /*switch*/ N<0>() );
    bench( xs, ys, ps, ds, /*simd*/ N<1>(), /*switch*/ N<1>() );
}
