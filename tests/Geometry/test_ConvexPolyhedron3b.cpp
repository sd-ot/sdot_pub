#include "../../src/sdot/Geometry/ConvexPolyhedron3b.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/P.h"
using namespace sdot;


template<class Pc,class Pts>
void test_cuts( Pts origs, Pts norms, VtkOutput &vo, int &cpt_vo ) {
    using Cp = ConvexPolyhedron3<Pc>;
    using TF = typename Cp::TF;
    using CI = typename Cp::CI;
    using Pt = typename Cp::Pt;

    // initial cell
    Cp lc;
    lc = typename Cp::Box{ { -2.0 }, { 2.0 } };

    // cuts
    std::vector<TF> cut_dx;
    std::vector<TF> cut_dy;
    std::vector<TF> cut_dz;
    std::vector<TF> cut_ps;
    std::vector<CI> cut_id;
    for( std::size_t i = 0; i < origs.size(); ++i ) {
        Pt N = normalized( norms[ i ] );
        cut_dx.push_back( N.x );
        cut_dy.push_back( N.y );
        cut_dz.push_back( N.z );
        cut_ps.push_back( dot( origs[ i ], N ) );
        cut_id.push_back( nullptr );
    }

    lc.plane_cut( { cut_dx.data(), cut_dy.data(), cut_dz.data() }, cut_ps.data(), cut_id.data(), cut_dx.size() );
    PN( lc );

    // display
    Pt off{ 1.5 * int( cpt_vo % 8 ), 1.5 * int( cpt_vo / 8 ), 0.0 };
    lc.display_vtk( vo, { TF( cpt_vo ) }, off );
    ++cpt_vo;

}

template<class Pc>
void test_sphere( VtkOutput &vo, int &cpt_vo ) {
    using Pt = Point3<double>;
    std::vector<Pt> origs, norms;
    for( std::size_t i = 0; i < 14; ++i ) {
        Pt p;
        for( int d = 0; d < 3; ++d )
            p[ d ] = 2.0 * rand() / RAND_MAX - 1;
        p /= norm_2( p );

        origs.push_back( p );
        norms.push_back( p );
    }
    test_cuts<Pc>( origs, norms, vo, cpt_vo );
}

int main() {
    struct Pc {
        enum { store_the_normals = false };
        enum { allow_ball_cut    = false };
        using  TF                = double;
        using  TI                = std::size_t;
        using  Pt                = Point3<TF>;
        struct Dirac             { TF weight; Pt pos; };

    };
    int cpt_vo = 0;
    VtkOutput vo( { "smurf" } );
    test_sphere<Pc>( vo, cpt_vo );
    vo.save( "vtk/pd.vtk" );
}
