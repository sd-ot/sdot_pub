#include "../../../src/sdot/Grids/LGrid.h"
#include "../../../src/sdot/Support/P.h"
#include <fstream>
using namespace sdot;

// // nsmake cxx_name clang++
// // nsmake cpp_flag -march=native

// // nsmake cpp_flag -ffast-math
// // nsmake cpp_flag -O3

template<class Grid>
void display( Grid &grid, std::string filename, std::string grid_filemane = {} ) {
    using Di = typename Grid::Dirac;
    using CP = typename Grid::CP;
    using TF = typename Grid::TF;
    std::mutex m;

    if ( grid_filemane.size() ) {
        VtkOutput vog;
        grid.display_vtk( vog, { .weight_elevation = 1 } );
        vog.save( grid_filemane );
    }

    if ( filename.size() ) {
        TF area = 0;
        VtkOutput voc( Di::names() );
        typename CP::Box box{ { TF( 0 ) }, { TF( 1 ) } };
        grid.for_each_laguerre_cell( [&]( auto &cp, auto &d, int ) {
            m.lock();

            cp.display_vtk( voc, d.values() );
            area += cp.integral();

            m.unlock();
        }, box );

        voc.save( filename );
        P( area );
    }
}

template<class TF>
struct MtVal {
    enum {          sep       = 16 };

    /**/            MtVal     ( TF val = 0 ) : values( sep * thread_pool.nb_threads(), val ) {}
    void            clear     () { for( std::size_t i = 0; i < values.size(); i += sep ) values[ i ] = 0; }

    TF             &operator[]( int num_thread ) { return values[ sep * num_thread ]; }
    TF              sum       () const { TF res = 0; for( std::size_t i = 0; i < values.size(); i += sep ) res += values[ i ]; return res; }

    std::vector<TF> values;
};

template<class Pc>
void generate( std::size_t nb_diracs ) {
    using Dirac = typename Pc::Dirac;
    using Grid  = LGrid<Pc>;
    using CP    = typename Grid::CP;
    using Pt    = typename Grid::Pt;
    using TF    = typename Grid::TF;
    using TI    = typename Grid::TI;

    // load
    std::size_t nb_voro_points = 20;
    std::vector<Dirac> diracs( nb_voro_points );
    for( std::size_t n = 0; n < nb_voro_points; ++n ) {
        for( std::size_t d = 0; d < Grid::dim; ++d )
            diracs[ n ].pos[ d ] = 1.0 * rand() / RAND_MAX;
        diracs[ n ].index = n;
    }

    // grid
    Grid grid( 20 );
    grid.construct( diracs.data(), diracs.size() );

    // solve init => first residual,
    // display( grid, "vtk/pd.vtk", "vtk/grid.vtk" );

    // sum of areas
    struct SA { Simplex<TF,Grid::dim,Grid::dim-1> simplex; TF acc; };
    typename CP::Box box{ { TF( 0 ) }, { TF( 1 ) } };
    std::vector<SA> sas;
    TF total_area = 0;
    std::mutex m;
    grid.for_each_laguerre_cell( [&]( const CP &cp, const Dirac &dirac, int /*num_thread*/ ) {
        m.lock();
        cp.for_each_boundary_item( [&]( const typename CP::BoundaryItem &face ) {
            if ( face.cut_id > &dirac ) {
                face.foreach_simplex( [&]( const auto &simplex ) {
                    total_area += simplex.mass();
                    sas.push_back( { simplex, total_area } );
                } );
            }
        } );
        m.unlock();
    }, box );

    //
    TI ind_simplex = 0;
    // VtkOutput voc;
    std::string name = va_string( "/data/sdot/faces_20p_{}D_{}.txt", int( Grid::dim ), nb_diracs );
    std::ofstream fout( name.c_str() );
    for( TF s = total_area / nb_diracs, a = 0.5 * s; a < total_area; a += s ) {
        while ( sas[ ind_simplex ].acc < a )
            ++ind_simplex;
        Pt p = sas[ ind_simplex ].simplex.random_point( []() { return TF( 1 ) * rand() / RAND_MAX; } );
        for( std::size_t d = 0; d < Grid::dim; ++d )
            p[ d ] += 1e-5 * rand() / RAND_MAX;
        fout << p << " 0\n";
        // voc.add_point(  );
    }
    // voc.save( "vtk/points.vtk" );
}

template<int d>
struct Pc {
    enum { store_the_normals = false };
    enum { allow_ball_cut    = false };
    enum { w_bounds_order    = 0     };
    enum { dim               = d     };
    using  TI                = std::uint64_t;
    using  TF                = double;
    using  Pt                = typename PointTraits<TF,d>::type;

    struct Dirac {
        static std::vector<std::string> names() { return { "weight", "index" }; }
        std::vector<TF> values() const { return { weight, TF( index ) }; }

        TF weight;
        TI index;
        Pt pos;
    };
};

int main() {
    for( std::size_t nb_diracs = 1e6; nb_diracs < 1e9; nb_diracs = nb_diracs * 16 / 10 )
        generate<Pc<2>>( nb_diracs );
}
