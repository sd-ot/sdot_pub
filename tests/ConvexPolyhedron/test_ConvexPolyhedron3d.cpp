#include "../../src/sdot/ConvexPolyhedron/ConvexPolyhedron.h"
#include "../../src/sdot/ConvexPolyhedron/display_vtk.h"
#include "../../src/sdot/Support/VtkOutput.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/P.h"
using namespace sdot;

template<class Cp>
void test_sphere() {
    using Pt = typename Cp::Pt;
    using TF = typename Cp::TF;

    Cp cp( Pt( -2.0 ), Pt( 2.0 ) );

    std::vector<TF> cx, cy, cz, cs, ci;
    for( std::size_t n = 0; n < 1000; ++n ) {
        double p = std::acos( 2.0 * rand() / RAND_MAX - 1.0 );
        double t = 2.0 * M_PI * rand() / RAND_MAX;
        cx.push_back( std::sin( p ) * std::cos( t ) );
        cy.push_back( std::sin( p ) * std::sin( t ) );
        cz.push_back( std::cos( p ) );
        cs.push_back( 1.0 );
        ci.push_back( 0.0 );
    }

    cp.plane_cut( { cx.data(), cy.data(), cz.data() }, cs.data(), ci.data(), cx.size() );
    // P( cp );

    VtkOutput vo;
    display_vtk( vo, cp );
    vo.save( "vtk/pd.vtk" );
}

int main() {
    struct Pc { enum { dim = 3 }; using CI = double; using TF = double; };

    test_sphere<ConvexPolyhedron<Pc>>();
}
