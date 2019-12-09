#include "../../src/sdot/Geometry/ConvexPolyhedron3b.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
#include <map>
using namespace sdot;

//// nsmake cxx_name clang++
//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O3
//// nsmake lib_flag -O3

struct Pc {
    enum { store_the_normals = false };
    enum { allow_ball_cut    = false };
    using  TF                = double;
    using  TI                = std::size_t;
    using  Pt                = Point3<TF>;

    struct Dirac {
        TF weight;
        Pt pos;
    };
};

using Cp = ConvexPolyhedron3<Pc>;
using TF = Cp::TF;
using Pt = Cp::Pt;

template<class Box>
void __attribute__ ((noinline)) set_box( Cp &cp, const Box &box, std::vector<TF> &/*xs*/, std::vector<TF> &/*ys*/, std::vector<TF> &/*zs*/, std::vector<TF> &/*ps*/, std::vector<Pc::Dirac *> &/*ds*/ ) {
    cp = box;
}

template<class Box>
void __attribute__ ((noinline)) cut_proc( Cp &cp, const Box &box, std::vector<TF> &xs, std::vector<TF> &ys, std::vector<TF> &zs, std::vector<TF> &ps, std::vector<Pc::Dirac *> &ds ) {
    cp = box;
    cp.plane_cut( { xs.data(), ys.data(), zs.data() }, ps.data(), ds.data(), xs.size(), N<0>() );
}

void bench( std::vector<TF> xs, std::vector<TF> ys, std::vector<TF> zs, std::vector<TF> ps, std::vector<Pc::Dirac *> ds, std::uint64_t nb_reps ) {
    Cp::Box box{ TF( -1 ), TF( 1 ) };
    std::uint64_t t0 = 0, t1 = 0;

    // overhead
    Cp cp;
    RDTSC_START( t0 );
    for( std::size_t rep = 0; rep < nb_reps; ++rep )
        set_box( cp, box, xs, ys, zs, ps, ds );
    RDTSC_FINAL( t1 );
    double dt_set_box = 1.0 * ( t1 - t0 ) / nb_reps;

    // cuts
    auto ti0 = time();
    RDTSC_START( t0 );
    for( std::size_t rep = 0; rep < nb_reps; ++rep )
        cut_proc( cp, box, xs, ys, zs, ps, ds );
    RDTSC_FINAL( t1 );
    P( t1 - t0, time() - ti0 );
    std::uint64_t dt_cut_proc = 1.0 * ( t1 - t0 ) / nb_reps;

    if ( nb_reps > 1 ) {
        P( dt_set_box );
        P( ( dt_cut_proc - dt_set_box ) / xs.size() );
    }
    PN( cp );

    VtkOutput vo;
    cp.display_vtk( vo );
    vo.save( "vtk/pd.vtk" );
}


int main() {
    bool single_test = 0;
    bool single_dir = 1; // single_test;

    std::vector<Pt> directions;
    if ( single_dir ) {
        directions.push_back( { 1, 0, 0 } );
    } else {
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
    }

    std::size_t nb_cuts = single_test ? 1 : 64;
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

    bench( xs, ys, zs, ps, ds, single_test ? 1 : 200000 );
}
