#include "../src/sdot/Geometry/ConvexPolyhedron3.h"
#include "../src/sdot/Support/StaticRange.h"
#include "../src/sdot/Support/ASSERT.h"
#include "../src/sdot/Support/Time.h"
#include "../src/sdot/Support/P.h"
#include "../src/sdot/VtkOutput.h"
#include <map>
using namespace sdot;

// // nsmake cpp_flag -march=native
//// nsmake cpp_flag -O2

template<class Cp>
void test_regular_cuts( VtkOutput &vo, int &cpt_vo, typename Cp::Pt n, typename Cp::TF d ) {
    using TF = typename Cp::TF;
    using CI = typename Cp::CI;
    using Pt = typename Cp::Pt;
    constexpr int flags = 0;
    n /= norm_2( n );

    // initial cell
    Cp lc( typename Cp::Box{ { 0, 0, 0 }, { 1, 1, 1 } } );

    // cuts
    std::vector<TF> cut_dx = { n[ 0 ] };
    std::vector<TF> cut_dy = { n[ 1 ] };
    std::vector<TF> cut_dz = { n[ 2 ] };
    std::vector<TF> cut_ps = { d      };
    std::vector<CI> cut_id = { 9      };

    lc.plane_cut( { cut_dx.data(), cut_dy.data(), cut_dz.data() }, cut_ps.data(), cut_id.data(), cut_dx.size(), N<flags>() );
    lc.check();

    // display
    Pt off{ 1.5 * int( cpt_vo % 8 ), 1.5 * int( cpt_vo / 8 ), 0.0 };
    lc.display( vo, { TF( cpt_vo ) }, off );
    ++cpt_vo;
}

template<class Cp>
void test_diam() {
    using TF = typename Cp::TF;
    using CI = typename Cp::CI;
    using Pt = typename Cp::Pt;
    constexpr int flags = 0;

    // initial cell
    Cp lc( typename Cp::Box{ { -2, -2, -2 }, { +2, +2, +2 } } );

    // cuts
    std::vector<TF> cut_dx, cut_dy, cut_dz, cut_ps;
    std::vector<CI> cut_id;
    for( int dz = -1, c = 0; dz <= 1; ++dz ) {
        for( int dy = -1; dy <= 1; ++dy ) {
            for( int dx = -1; dx <= 1; ++dx ) {
                if ( dx || dy || dz ) {
                    Pt n( dx, dy, dz );
                    n /= norm_2( n );

                    cut_dx.push_back( n.x );
                    cut_dy.push_back( n.y );
                    cut_dz.push_back( n.z );
                    cut_ps.push_back( 1.0 );
                    cut_id.push_back( c++ );
                }
            }
        }
    }

    lc.plane_cut( { cut_dx.data(), cut_dy.data(), cut_dz.data() }, cut_ps.data(), cut_id.data(), 1 /*cut_dx.size()*/, N<flags>() );
    lc.check();

    // display
    VtkOutput vo( { "smurf" } );
    lc.display( vo );
    vo.save( "vtk/pd.vtk" );
}

template<class Cp>
void test_regular_cuts() {
    int cpt_vo = 0;
    VtkOutput vo( { "smurf" } );

    test_regular_cuts<Cp>( vo, cpt_vo, { +1, 0, 0 }, +0.2 );
    test_regular_cuts<Cp>( vo, cpt_vo, { 0, +1, 0 }, +0.2 );
    test_regular_cuts<Cp>( vo, cpt_vo, { 0, 0, +1 }, +0.2 );
    test_regular_cuts<Cp>( vo, cpt_vo, { -1, 0, 0 }, -0.2 );
    test_regular_cuts<Cp>( vo, cpt_vo, { 0, -1, 0 }, -0.2 );
    test_regular_cuts<Cp>( vo, cpt_vo, { 0, 0, -1 }, -0.2 );

    test_regular_cuts<Cp>( vo, cpt_vo, { 1, 1, 1 }, 0.9 * std::sqrt( 3.0 ) );
    test_regular_cuts<Cp>( vo, cpt_vo, { 1, 1, 1 }, 0.1 * std::sqrt( 3.0 ) );

    vo.save( "vtk/pd.vtk" );
}

int main() {
    struct Pc {
        enum { store_the_normals = false };
        enum { allow_ball_cut    = false };
        enum { block_size        = 64    };
        enum { dim               = 3     };
        using  TF                = double;
        using  TI                = std::size_t;
        using  CI                = std::size_t;

    };
    using Cp = ConvexPolyhedron3<Pc>;
    // test_regular_cuts<Cp>();
    test_diam<Cp>();
}
