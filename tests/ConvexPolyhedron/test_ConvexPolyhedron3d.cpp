#include "../../src/sdot/ConvexPolyhedron/ConvexPolyhedron.h"
#include "../../src/sdot/ConvexPolyhedron/display_vtk.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/P.h"
using namespace sdot;

template<class Cp>
void test_sphere( VtkOutput &vo, double off ) {
    using Pt = typename Cp::Pt;
    using TF = typename Cp::TF;

    Cp cp( Pt( -2.0 ), Pt( 2.0 ) );

    srand( 0 );
    std::vector<TF> cx, cy, cz, cs;
    for( std::size_t n = 0; n < 80; ++n ) {
        double p = std::acos( 2.0 * rand() / RAND_MAX - 1.0 );
        double t = 2.0 * M_PI * rand() / RAND_MAX;
        cx.push_back( std::sin( p ) * std::cos( t ) );
        cy.push_back( std::sin( p ) * std::sin( t ) );
        cz.push_back( std::cos( p ) );
        cs.push_back( 1.0 );
    }

    cp.plane_cut( { cx.data(), cy.data(), cz.data() }, cs.data(), cs.data(), cx.size(), N<0>(), [&]( auto &cp ) {
        display_vtk( vo, cp, { .offset = Pt{ off, 0, 0 } } );
    } );
}

int main() {
    struct Pc { enum { dim = 3 }; using CI = double; using TF = double; };
    VtkOutput vo;

    test_sphere<ConvexPolyhedron<Pc>>( vo, 0.0 );
    test_sphere<ConvexPolyhedron<Pc,3,ConvexPolyhedronOpt::Lt64>>( vo, 2.5 );

    vo.save( "vtk/pd.vtk" );
}
