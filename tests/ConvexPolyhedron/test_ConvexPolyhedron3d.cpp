#include "../../src/sdot/ConvexPolyhedron/ConvexPolyhedron.h"
#include "../../src/sdot/ConvexPolyhedron/display_vtk.h"
#include "../../src/sdot/Support/VtkOutput.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/P.h"
using namespace sdot;

template<class Cp>
void cut_and_disp( Cp &cp, std::array<const double *,3> cn, const double *cd, const double *ci, std::size_t nb_cuts ) {
    bool direct = true;
    cp.plane_cut( cn, cd, ci, nb_cuts, N<0>(), [&]( auto &cp, auto cn, auto cd, auto ci, std::size_t nb_cuts ) {
        cut_and_disp( cp, cn, cd, ci, nb_cuts );
        direct = false;
    } );

    if ( direct ) {
        VtkOutput vo;
        display_vtk( vo, cp );
        vo.save( "vtk/pd.vtk" );
    }
}

template<class Cp>
void test_sphere() {
    using Pt = typename Cp::Pt;
    using TF = typename Cp::TF;

    Cp cp( Pt( -2.0 ), Pt( 2.0 ) );

    std::vector<TF> cx, cy, cz, cs;
    for( std::size_t n = 0; n < 1000; ++n ) {
        double p = std::acos( 2.0 * rand() / RAND_MAX - 1.0 );
        double t = 2.0 * M_PI * rand() / RAND_MAX;
        cx.push_back( std::sin( p ) * std::cos( t ) );
        cy.push_back( std::sin( p ) * std::sin( t ) );
        cz.push_back( std::cos( p ) );
        cs.push_back( 1.0 );
    }

    cut_and_disp( cp, { cx.data(), cy.data(), cz.data() }, cs.data(), cs.data(), cx.size() );
}

int main() {
    struct Pc { enum { dim = 3 }; using CI = double; using TF = double; };

    test_sphere<ConvexPolyhedron<Pc>>();
}
