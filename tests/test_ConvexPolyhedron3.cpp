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
void test_regular_cuts( VtkOutput &vo, int &cpt_vo ) {
    using TF = typename Cp::TF;
    using CI = typename Cp::CI;
    using Pt = typename Cp::Pt;
    constexpr int flags = 0;

    // initial cell
    Cp lc( typename Cp::Box{ { 0, 0, 0 }, { 1, 1, 1 } } );
    // lc.write_to_stream( std::cout, 1 );

    // cuts
    std::vector<TF> cut_dx, cut_dy, cut_dz, cut_ps;
    std::vector<CI> cut_id;
    //    for( int dz = -1, c = 0; dz <= 1; ++dz ) {
    //        for( int dy = -1; dy <= 1; ++dy ) {
    //            for( int dx = -1; dx <= 1; ++dx ) {
    //                if ( dx || dy || dz ) {
    //                    Pt n( dx, dy, dz );
    //                    n /= norm_2( n );
    //                    cut_dx.push_back( n.x );
    //                    cut_dy.push_back( n.y );
    //                    cut_dz.push_back( n.z );
    //                    cut_ps.push_back( 1.0 );
    //                    cut_id.push_back( c++ );
    //                }
    //            }
    //        }
    //    }

    cut_dx.push_back( 1 );
    cut_dy.push_back( 0 );
    cut_dz.push_back( 0 );
    cut_ps.push_back( 0.5 );
    cut_id.push_back( 9 );

    lc.plane_cut( { cut_dx.data(), cut_dy.data(), cut_dz.data() }, cut_ps.data(), cut_id.data(), cut_dx.size(), N<flags>() );

    // display
    Pt off{ 4.5 * TF( cpt_vo % 8 ), 4.5 * TF( cpt_vo / 8 ), 0 };
    ++cpt_vo;

    lc.display( vo, { TF( cpt_vo ) }, off );

    // ASSERT( lc.nb_nodes() == nb_nodes, "" );
}

template<class Cp>
void test_regular_cuts() {
    int cpt_vo = 0;
    VtkOutput vo( { "smurf" } );

    test_regular_cuts<Cp>( vo, cpt_vo );

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
    test_regular_cuts<Cp>();
}
