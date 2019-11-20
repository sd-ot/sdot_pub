#include "../../src/sdot/Geometry/ConvexPolyhedron3.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
#include <map>
using namespace sdot;

//// nsmake cxx_name clang++
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
    using  Pt                = Point3<TF>;

    struct Dirac {
        TF weight;
        Pt pos;
    };
};

using Cp = ConvexPolyhedron3<Pc>;
using TF = Cp::TF;
using Pt = Cp::Pt;

void bench( std::vector<TF> xs, std::vector<TF> ys, std::vector<TF> zs, std::vector<TF> ps, std::vector<Pc::Dirac *> ds ) {
    constexpr int flags = 0;

    // overhead
    Cp cp;
    TF sum = 0;
    Cp::Box box{ { -1, -1, -1 }, { 1, 1, 1 } };
    std::uint64_t t0 = 0, t1 = 0, nb_reps = 128000;
    RDTSC_START( t0 );
    for( std::size_t rep = 0; rep < nb_reps; ++rep )
        cp = box;
    RDTSC_FINAL( t1 );
    std::uint64_t overhead = t1 - t0;

    // cuts
    RDTSC_START( t0 );
    for( std::size_t rep = 0; rep < nb_reps; ++rep ) {
        cp = box;
        cp.plane_cut( { xs.data(), ys.data(), zs.data() }, ps.data(), ds.data(), xs.size(), N<flags>() );
        sum += cp.nb_nodes();
    }
    RDTSC_FINAL( t1 );
    std::uint64_t dt = ( t1 - t0 - overhead ) / ( nb_reps * xs.size() );

    // P( cp );
    P( sum, overhead, t1 - t0, dt );

    VtkOutput vo;
    cp.display_vtk( vo );
    vo.save( "vtk/pd.vtk" );
}


int main() {
    std::vector<Pt> directions;
    for( TF z = -1; z <= 1; ++z ) {
        for( TF y = -1; y <= 1; ++y ) {
            for( TF x = -1; x <= 1; ++x ) {
                if ( x || y || z ) {
                    Pt p( x, y, z );
                    directions.push_back( p / norm_2( p ) );
                }
            }
        }
    }

    std::size_t nb_cuts = 120;
    std::vector<TF> xs, ys, zs, ps;
    std::vector<Pc::Dirac *> ds;
    for( std::size_t n = 0; n < nb_cuts; ++n ) {
        int th = random() % directions.size();
        xs.push_back( directions[ th ].x );
        ys.push_back( directions[ th ].y );
        zs.push_back( directions[ th ].z );
        ps.push_back( 10.0 / ( 11 + n ) + ( n % 2 ) );
        ds.push_back( nullptr );
    }

    bench( xs, ys, zs, ps, ds );
}
