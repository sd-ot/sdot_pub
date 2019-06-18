#include <vector>
static std::vector<int> bc( 4096, 0 );

#include "../src/sdot/Geometry/ConvexPolyhedron2.h"
#include "../src/sdot/Support/StaticRange.h"
#include "../src/sdot/Support/ASSERT.h"
#include "../src/sdot/Support/Time.h"
#include "../src/sdot/Support/P.h"
#include "../src/sdot/VtkOutput.h"
#include <map>
using namespace sdot;

// // nsmake cpp_flag -march=skylake
// // nsmake cxx_name clang++

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O2


template<class Cp,int Simd,int Switch>
void test( VtkOutput &vo, int &cpt_vo, std::size_t nb_nodes, N<Simd>, N<Switch> ) {
    using Pt = typename Cp::Pt;
    using TF = typename Cp::TF;

    // initial cell
    Cp lc( typename Cp::Box{ { -2, -2 }, { +2, +2 } } );
    constexpr int flags = ConvexPolyhedron::do_not_use_simd     * ( Simd   == 0 ) +
                          ConvexPolyhedron::do_not_use_switches * ( Switch == 0 );
    for( std::size_t i = 0; i < nb_nodes; ++i ) {
        TF a = i * 2 * M_PI / nb_nodes;
        Pt n( cos( a ), sin( a ) );
        lc.plane_cut( n, n, 17, N<flags>() );
    }

    lc.display( vo, { TF( cpt_vo ) }, { 2.5 * TF( cpt_vo % 8 ), 2.5 * TF( cpt_vo / 8 ) } );
    ++cpt_vo;;
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

    int cpt_vo = 0;
    VtkOutput vo( { "smurf" } );

    for( std::size_t nb_nodes = 3; nb_nodes < 11; ++nb_nodes )
        test<Cp>( vo, cpt_vo, nb_nodes, /*simd*/ N<0>(), /*switch*/ N<0>() );
    for( std::size_t nb_nodes = 3; nb_nodes < 11; ++nb_nodes )
        test<Cp>( vo, cpt_vo, nb_nodes, /*simd*/ N<1>(), /*switch*/ N<0>() );
    for( std::size_t nb_nodes = 3; nb_nodes < 11; ++nb_nodes )
        test<Cp>( vo, cpt_vo, nb_nodes, /*simd*/ N<1>(), /*switch*/ N<1>() );

    vo.save( "vtk/pd.vtk" );
}
