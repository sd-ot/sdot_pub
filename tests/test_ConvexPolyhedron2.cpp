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
void test_regular_cuts( VtkOutput &vo, int &cpt_vo, std::size_t nb_nodes, N<Simd>, N<Switch> ) {
    using Cut = typename Cp::Cut;
    using TF = typename Cp::TF;

    // initial cell
    Cp lc( typename Cp::Box{ { -2, -2 }, { +2, +2 } } );
    constexpr int flags = ConvexPolyhedron::do_not_use_simd     * ( Simd   == 0 ) +
                          ConvexPolyhedron::do_not_use_switch * ( Switch == 0 );

    // cuts
    for( std::size_t i = 0; i < nb_nodes; ++i ) {
        TF a = i * 2 * M_PI / nb_nodes;
        Cut cut{ { cos( a ), sin( a ) }, 1.0, 17 };
        lc.plane_cut( &cut, 1, N<flags>() );
    }

    // check
    lc.display( vo, { TF( cpt_vo ) }, { 4.5 * TF( cpt_vo % 8 ), 4.5 * TF( cpt_vo / 8 ) } );
    ASSERT( lc.nb_nodes() == nb_nodes, "" );
    ++cpt_vo;
}

template<class Cp>
void test_regular_cuts() {
    int cpt_vo = 0;
    VtkOutput vo( { "smurf" } );

    for( std::size_t nb_nodes = 3; nb_nodes < 11; ++nb_nodes )
        test_regular_cuts<Cp>( vo, cpt_vo, nb_nodes, /*simd*/ N<0>(), /*switch*/ N<0>() );
    for( std::size_t nb_nodes = 3; nb_nodes < 11; ++nb_nodes )
        test_regular_cuts<Cp>( vo, cpt_vo, nb_nodes, /*simd*/ N<1>(), /*switch*/ N<0>() );
    for( std::size_t nb_nodes = 3; nb_nodes < 11; ++nb_nodes )
        test_regular_cuts<Cp>( vo, cpt_vo, nb_nodes, /*simd*/ N<1>(), /*switch*/ N<1>() );

    vo.save( "vtk/pd.vtk" );
}

template<class Cp,int Simd,int Switch>
void test_measure( VtkOutput &vo, int &cpt_vo, const std::vector<typename Cp::Pt> &positions, N<Simd>, N<Switch> ) {
    constexpr int flags = ConvexPolyhedron::do_not_use_simd   * ( Simd   == 0 ) +
                          ConvexPolyhedron::do_not_use_switch * ( Switch == 0 );
    using Cut = typename Cp::Cut;
    using TF = typename Cp::TF;
    using Pt = typename Cp::Pt;

    //
    TF volume = 0;
    for( std::size_t i = 0; i < positions.size(); ++i ) {
        Cp lc( typename Cp::Box{ { 0, 0 }, { 1, 1 } } );
        for( std::size_t j = 0; j < positions.size(); ++j ) {
            if ( i == j )
                continue;
            Pt mid = 0.5 * ( positions[ i ] + positions[ j ] );
            Pt dir = positions[ j ] - positions[ i ];
            Cut cut{ dir, dot( dir, mid ), i };
            lc.plane_cut( &cut, 1, N<flags>() );
        }
        lc.display( vo, { TF( i ) }, { 1.5 * TF( cpt_vo % 8 ), 1.5 * TF( cpt_vo / 8 ) } );
        volume += lc.integral();
    }
    ++cpt_vo;

    P( volume );
}

template<class Cp>
void test_measure( std::size_t nb_nodes = 500 ) {
    int cpt_vo = 0;
    VtkOutput vo( { "smurf" } );

    std::vector<typename Cp::Pt> positions;
    for( std::size_t i = 0; i < nb_nodes; ++i )
        positions.push_back( { 1.0 * rand() / RAND_MAX, 1.0 * rand() / RAND_MAX } );

    test_measure<Cp>( vo, cpt_vo, positions, /*simd*/ N<0>(), /*switch*/ N<0>() );
    test_measure<Cp>( vo, cpt_vo, positions, /*simd*/ N<1>(), /*switch*/ N<0>() );
    test_measure<Cp>( vo, cpt_vo, positions, /*simd*/ N<1>(), /*switch*/ N<1>() );

    vo.save( "vtk/measure.vtk" );
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
    //    test_regular_cuts<Cp>();
    test_measure<Cp>();
}
