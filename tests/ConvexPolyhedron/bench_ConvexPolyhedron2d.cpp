#include "../../src/sdot/ConvexPolyhedron/ConvexPolyhedron.h"
#include "../../src/sdot/ConvexPolyhedron/display_vtk.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
using namespace sdot;

//// nsmake cxx_name clang++
//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O3


struct Pc { using CI = double; using TF = double; };
using Cp = ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>;
using Pt = typename Cp::Pt;
using TF = typename Cp::TF;


template<class Box>
void __attribute__ ((noinline)) set_box( Cp &cp, const Box &box, std::vector<TF> &/*xs*/, std::vector<TF> &/*ys*/, std::vector<TF> &/*ps*/, std::vector<TF> &/*ds*/ ) {
    cp = box;
}

template<class Box>
void __attribute__ ((noinline)) cut_proc( Cp &cp, const Box &box, std::vector<TF> &xs, std::vector<TF> &ys, std::vector<TF> &ps, std::vector<TF> &ds ) {
    cp = box;
    cp.plane_cut( { xs.data(), ys.data() }, ps.data(), ds.data(), xs.size() );
}

void bench( std::size_t nb_reps ) {
    std::size_t nb_cuts = 120;
    std::vector<TF> xs, ys, ps, ds;
    for( std::size_t n = 0; n < nb_cuts; ++n ) {
        TF th = ( random() % 8 ) * M_PI / 5;
        ds.push_back( 0.0 );
        xs.push_back( cos( th ) );
        ys.push_back( sin( th ) );
        ps.push_back( 10.0 / ( 11 + n ) + ( n % 2 ) );
    }

    Cp box{ TF( -1 ), TF( 1 ) };
    std::uint64_t t0 = 0, t1 = 0;

    // overhead
    Cp cp;
    RDTSC_START( t0 );
    for( std::size_t rep = 0; rep < nb_reps; ++rep )
        set_box( cp, box, xs, ys, ps, ds );
    RDTSC_FINAL( t1 );
    double dt_set_box = 1.0 * ( t1 - t0 ) / nb_reps;

    // cuts
    auto ti0 = time();
    RDTSC_START( t0 );
    for( std::size_t rep = 0; rep < nb_reps; ++rep )
        cut_proc( cp, box, xs, ys, ps, ds );
    RDTSC_FINAL( t1 );
    auto ti1 = time();
    double dt_cut_proc = 1.0 * ( t1 - t0 ) / nb_reps;
    double nb_cycles_per_cut = ( dt_cut_proc - dt_set_box ) / xs.size();

    P( ti1 - ti0, nb_cycles_per_cut );
}

int main( int argc, char **argv ) {
    bench( argc > 1 ? atoi( argv[ 1 ] ) : 800000 );
}
