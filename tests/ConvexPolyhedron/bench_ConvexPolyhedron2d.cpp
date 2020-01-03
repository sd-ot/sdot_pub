#include "../../src/sdot/ConvexPolyhedron/ConvexPolyhedron.h"
#include "../../src/sdot/ConvexPolyhedron/display_vtk.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
using namespace sdot;

// // nsmake cxx_name clang++
//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O3


struct Pc { using CI = std::size_t; using TF = double; };
using Cp = ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>;
using Pt = typename Cp::Pt;
using TF = typename Cp::TF;
using CI = typename Cp::CI;


template<class Box>
void __attribute__ ((noinline)) set_box( Cp &cp, const Box &box, std::vector<TF> &/*xs*/, std::vector<TF> &/*ys*/, std::vector<TF> &/*ps*/, std::vector<CI> &/*ds*/ ) {
    cp = box;
}

template<class Box>
void __attribute__ ((noinline)) cut_proc( Cp &cp, const Box &box, std::vector<TF> &xs, std::vector<TF> &ys, std::vector<TF> &ps, std::vector<CI> &ds ) {
    cp = box;
    cp.plane_cut( { xs.data(), ys.data() }, ps.data(), ds.data(), xs.size() );
}

void bench( std::size_t nb_reps = 5000 ) {
    std::size_t nb_cuts = 120;
    std::vector<CI> ds;
    std::vector<TF> xs, ys, ps;
    for( std::size_t n = 0; n < nb_cuts; ++n ) {
        TF th = ( random() % 8 ) * M_PI / 5;
        ds.push_back( 0.0 );
        xs.push_back( cos( th ) );
        ys.push_back( sin( th ) );
        ps.push_back( 10.0 / ( 11 + n ) + ( n % 2 ) );
    }

    Cp box{ TF( -1 ), TF( 1 ) };

    // overhead
    Cp cp;
    std::uint64_t overhead = -1ul;
    for( std::size_t rep = 0; rep < nb_reps; ++rep ) {
        std::uint64_t t0 = 0, t1 = 0;
        RDTSC_START( t0 );
        set_box( cp, box, xs, ys, ps, ds );
        RDTSC_FINAL( t1 );
        overhead = std::min( overhead, t1 - t0 );
    }

    // cuts
    std::uint64_t compute = -1ul;
    for( std::size_t rep = 0; rep < nb_reps; ++rep ) {
        std::uint64_t t0 = 0, t1 = 0;
        RDTSC_START( t0 );
        for( std::size_t rep = 0; rep < nb_reps; ++rep )
            cut_proc( cp, box, xs, ys, ps, ds );
        RDTSC_FINAL( t1 );
        compute = std::min( compute, t1 - t0 );
    }

    double nb_cycles_per_cut = 1.0 * ( compute - overhead ) / xs.size();

    P( cp );
    P( nb_cycles_per_cut );
}

int main( int /*argc*/, char **/*argv*/ ) {
    bench();
}
