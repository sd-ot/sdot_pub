#include "../../src/sdot/Support/StaticRange.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
#include "../../src/sdot/ZGrid.h"
#include <cnpy.h>
#include <map>

#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

using namespace sdot;

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O2

template<int _dim>
struct Pc {
    enum { store_the_normals = false };
    enum { allow_ball_cut    = false };
    enum { dim               = _dim };
    using  TF                = double;
    using  TI                = std::size_t;
    using  CI                = std::size_t;
};

template<class Pc>
struct Boundary {
    using                Grid           = ZGrid<Pc>;
    static constexpr int dim            = Pc::dim;
    using                CP             = typename Grid::CP;
    using                Pt             = typename Grid::Pt;
    using                TF             = typename Grid::TF;
    using                TI             = typename Grid::TI;

    /**/                 Boundary       ( std::array<Pt,dim> points ) : points( points ) { make_measure(); }

    void                 write_to_stream( std::ostream &os ) const { os << points; }
    void                 make_measure   () { if ( dim == 2 ) measure = norm_2( points[ 1 ] - points[ 0 ] ); else TODO; }
    Pt                   random_point   () const { TF a = 1.0 * rand() / RAND_MAX; return ( 1 - a ) * points[ 0 ] + a * points[ 1 ]; }

    TF                   measure;
    std::array<Pt,dim>   points;
    TF                   acc;
};

template<class Pc>
void test( std::size_t nb_diracs, std::size_t nb_points ) {
    static constexpr int dim = Pc::dim;
    using Grid = ZGrid<Pc>;
    using CP = typename Grid::CP;
    using Pt = typename Grid::Pt;
    using TF = typename Grid::TF;
    using TI = typename Grid::TI;

    std::vector<TF> positions_x( nb_diracs );
    std::vector<TF> positions_y( nb_diracs );
    std::vector<TF> weights( nb_diracs, 0 );
    for( std::size_t i = 0; i < nb_diracs; ++i ) {
        positions_x[ i ] = 1.0 * rand() / RAND_MAX;
        positions_y[ i ] = 1.0 * rand() / RAND_MAX;
    }

    Grid grid;
    grid.update( { positions_x.data(), positions_y.data() }, weights.data(), nb_diracs, N<Grid::homogeneous_weights>() );

    CP b( typename CP::Box{ { 0, 0 }, { 1, 1 } } );
    std::mutex m;
    std::vector<Boundary<Pc>> boundaries;
    grid.for_each_laguerre_cell( [&]( CP &cp, std::size_t num, int /*num_thread*/ ) {
        std::vector<std::array<Pt,dim>> boundaries_as_simplex;
        cp.for_each_boundary_item( [&]( const typename CP::BoundaryItem &boundary_item ) {
            if ( boundary_item.id != -1ul && boundary_item.id > num )
                boundary_item.add_simplex_list( boundaries_as_simplex );
        } );

        m.lock();
        for( const auto &pts : boundaries_as_simplex )
            boundaries.push_back( pts );
        m.unlock();
    }, b, { positions_x.data(), positions_y.data() }, weights.data(), nb_diracs, N<Grid::homogeneous_weights>() );

    // P( boundaries );

    TF acc = 0;
    for( Boundary<Pc> &b : boundaries ) {
        acc += b.measure;
        b.acc = acc;
    }
    for( Boundary<Pc> &b : boundaries )
        b.acc /= acc;

    boost::mt19937 rng; // not seeded (it's not relevant)
    boost::normal_distribution<> nd( 0.0, 0.001 );
    boost::variate_generator<boost::mt19937&,boost::normal_distribution<>> var_nor(rng, nd);

    std::vector<Pt> points;
    for( TI n = 0, i = 0; n < nb_points; ++n ) {
        TF a = ( n + 0.5 ) / nb_points;
        while ( boundaries[ i ].acc < a )
            ++i;
        Pt dx;
        for( std::size_t d = 0; d < dim; ++d )
            dx[ d ] = var_nor();
        points.push_back( boundaries[ i ].random_point() + dx );
    }

    for( Pt p : points )
        std::cout << "             ( " << p.x << ", " << p.y << " )\n";

    VtkOutput vo;
    for( Pt p : points ) {
        Pt d( 0.001 );
        vo.add_lines( { p - d, p + d } );
        vo.add_lines( { p - rot90( d ), p + rot90( d ) } );
    }
    vo.save( "vtk/voro.vtk" );
}

int main() {
    test<Pc<2>>( 20, 500 );
}
