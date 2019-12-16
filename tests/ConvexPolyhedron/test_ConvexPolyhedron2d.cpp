#include "../../src/sdot/ConvexPolyhedron/ConvexPolyhedron.h"
#include "../../src/sdot/ConvexPolyhedron/display_vtk.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/P.h"
using namespace sdot;

void get_rand_perm_iota( std::function<void(std::vector<int>)> f, std::size_t m, int nb_real = 50 ) {
    std::vector<int> res;
    for( std::size_t i = 0; i < m; ++i )
        res.push_back( i );
    f( res );

    while( nb_real-- ) {
        for( std::size_t i = 0; i < m; ++i )
            std::swap( res[ i ], res[ i + rand() % ( m - i ) ] );
        f( res );
    }
}

template<class Cp>
void test_disc( VtkOutput &vo, std::size_t m, double off ) {
    using Pt = typename Cp::Pt;
    using TF = typename Cp::TF;

    int cpt = 0;
    get_rand_perm_iota( [&]( std::vector<int> perm ) {
        Cp cp( Pt( -2.0 ), Pt( 2.0 ), 17 );

        // srand( 0 );
        std::vector<TF> cx, cy, cs, ci;
        for( std::size_t n = 0; n < m; ++n ) {
            double t = 2.0 * M_PI * perm[ n ] / m; // rand() / RAND_MAX;
            cx.push_back( std::cos( t ) );
            cy.push_back( std::sin( t ) );
            ci.push_back( perm[ n ] );
            cs.push_back( 1.0 );
        }

        cp.plane_cut( { cx.data(), cy.data() }, cs.data(), ci.data(), cx.size(), N<0>(), [&]( auto &cp ) {
            if ( cpt++ == 0 )
                display_vtk( vo, cp, { .offset = { off, 0, 0 } } );

            cp.for_each_bound( [&]( const auto &bound ) {
                bound.for_each_simplex( [&]( auto simplex ) {
                    Pt c = simplex.centroid();
                    int a = std::round( std::atan2( c[ 1 ], c[ 0 ] ) * m / ( 2.0 * M_PI ) + m );
                    ASSERT( std::abs( norm_2( c ) - 1 ) < 1e-14, "..." );
                    ASSERT( a % m == bound.cut_id(), "..." );
                } );
            } );
        } );
    }, m );
}

int main() {
    struct Pc { using CI = double; using TF = double; };
    VtkOutput vo;

    double off = 0;
    for( int nb_nodes : { 4, 5, 6, 7, 8, 9, 10, 30, 100, 200 } ) {
        test_disc<ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64>>( vo, nb_nodes, off );
        //    test_sphere<ConvexPolyhedron<Pc>>( vo, 0.0 );
        off += 2.5;
    }

    vo.save( "vtk/pd.vtk" );
}
