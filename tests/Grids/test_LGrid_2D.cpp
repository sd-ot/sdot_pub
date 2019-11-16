#include "../../src/sdot/Grids/LGrid.h"
#include "../../src/sdot/Support/P.h"
using namespace sdot;

// // nsmake cxx_name clang++

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O3

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
        CP ic( typename CP::Box{ { 0, 0 }, { 1, 1 } }, nullptr );
        grid.for_each_laguerre_cell( [&]( auto &cp, auto &d, int ) {
            m.lock();

            cp.display_vtk( voc, d.values() );
            area += cp.integral();

            m.unlock();
        }, ic );

        voc.save( filename );
        //        P( area );
    }
}

template<class CP>
typename CP::TF for_each_cp_der( CP &cp, typename CP::Dirac &d0, const std::function<void( typename CP::Dirac &d1, typename CP::TF der_1 )> &f ) {
    using Dirac = typename CP::Dirac;
    using TF = typename CP::TF;
    constexpr TF coeff = 0.5;

    // derivative
    TF der_0 = 0;
    cp.for_each_boundary_measure( [&]( TF boundary_measure, Dirac *d1 ) {
        if ( d1 == nullptr )
            return;
        if ( d1 == &d0 ) {
            der_0 += coeff * boundary_measure / sqrt( d0.weight );
        } else {
            TF b_der = coeff * boundary_measure / norm_2( d0.pos - d1->pos );
            f( *d1, - b_der );
            der_0 += b_der;
        }
    }, d0.weight );

    // der_0 += cp.integration_der_wrt_weight( radial_func.func_for_final_cp_integration(), d0_weight );
    return der_0;
}

template<class TF>
struct MtVal {
    /**/            MtVal     ( TF val = 0 ) : values( thread_pool.nb_threads(), val ) {}
    void            clear     () { for( TF &v : values ) v = 0; }

    TF             &operator[]( int num_thread ) { return values[ num_thread ]; }
    TF              sum       () const { TF res = 0; for( TF v : values ) res += v; return res; }

    std::vector<TF> values;
};

template<class Pc>
void test_with_Pc() {
    //    constexpr int flags = 0;
    using         Dirac = typename Pc::Dirac;
    using         Grid  = LGrid<Pc>;
    constexpr int dim   = Pc::dim;
    using         CP    = typename Grid::CP;
    using         Pt    = typename Grid::Pt;
    using         TF    = typename Grid::TF;
    using         TI    = typename Grid::TI;

    using std::sqrt;

    // load
    std::size_t nb_diracs = 19;
    std::vector<Dirac> diracs( nb_diracs );
    for( std::size_t n = 0; n < nb_diracs; ++n ) {
        for( std::size_t d = 0; d < dim; ++d )
            diracs[ n ].pos[ d ] = 0.2 + 0.6 * rand() / RAND_MAX;
        diracs[ n ].weight = 0 * sin( diracs[ n ].pos.x ) * sin( diracs[ n ].pos.y );
        diracs[ n ].index = n;
    }

    // grid
    Grid grid( 20 );
    grid.construct( diracs.data(), diracs.size() );

    // solve init => first residual,
    //   # un premier parcourt pour trouver le M, le z, mettre 0 dans p, et faire le produit scalaire h
    //   r = B # résidu =
    //   z = M @ r
    //   p = z
    TF target_mass = TF( 1 ) / nb_diracs;
    CP ic( typename CP::Box{ { 0, 0 }, { 1, 1 } }, nullptr );
    grid.for_each_laguerre_cell( [&]( CP &cp, Dirac &d0, int /*num_thread*/ ) {
        TF der_0 = for_each_cp_der( cp, d0, [&]( Dirac &/*d1*/, TF /*der_1*/ ) {} );
        d0.r = target_mass - cp.integral();
        d0.z = d0.r / der_0;
        d0.p = d0.z;
    }, ic );

    //
    for( std::size_t num_iter = 0; num_iter < 100; ++num_iter ) {
        display( grid, va_string( "vtk/pd_{}.vtk", num_iter ), "vtk/grid.vtk" );

        // # calcul de q + np.dot( p, q )
        // h = np.dot( r, z )
        // q = A @ p
        MtVal<TF> dqp, ha;
        grid.for_each_laguerre_cell( [&]( CP &cp, Dirac &d0, int num_thread ) {
            // A @ p
            d0.q = 0;
            TF der_0 = for_each_cp_der( cp, d0, [&]( Dirac &d1, TF der_1 ) { d0.q += der_1 * d1.p; } );
            d0.q += der_0 * d0.p;

            dqp[ num_thread ] += d0.q * d0.p;
            ha[ num_thread ] += d0.r * d0.z;
        }, ic );

        // maj du poids
        TF alpha = ha.sum() / dqp.sum();
        grid.for_each_dirac( [&]( Dirac &d0, int /*num_thread*/ ) {
            d0.weight += alpha * d0.p;
        } );
        while ( true ) {
            bool ok = true;
            MtVal<TF> error, hb;
            grid.for_each_laguerre_cell( [&]( CP &cp, Dirac &d0, int num_thread ) {
                TF der_0 = for_each_cp_der( cp, d0, [&]( Dirac &d1, TF der_1 ) { d0.q += der_1 * d1.p; } );
                d0.r = target_mass - cp.integral();
                if ( der_0 )
                    d0.z = d0.r / der_0;
                else
                    ok = false;

                error[ num_thread ] += d0.r * d0.r;
                hb[ num_thread ] += d0.r * d0.z;
            }, ic );

            if ( ok ) {
                P( sqrt( error.sum() ) );

                // p = z + beta * p
                TF beta = hb.sum() / ha.sum();
                grid.for_each_dirac( [&]( Dirac &d0, int ) {
                    d0.p = d0.z + beta * d0.p;
                } );
                break;
            }

            P( "bim" );
            alpha *= TF( 0.5 );
            grid.for_each_dirac( [&]( Dirac &d0, int /*num_thread*/ ) {
                d0.weight -= alpha * d0.p;
            } );
        }
    }
}



int main() {
    struct Pc {
        enum { store_the_normals = false };
        enum { allow_ball_cut    = false };
        enum { w_bounds_order    = 0     };
        enum { dim               = 2     };
        using  TI                = std::uint64_t;
        using  TF                = double;
        using  Pt                = Point2<TF>;

        struct Dirac {
            static std::vector<std::string> names() { return { "weight", "index", "r", "p", "z", "q" }; }
            std::vector<TF> values() const { return { weight, TF( index ), r, p, z, q }; }

            //
            TF weight;
            TI index;
            Pt pos;

            // conjugate gradient
            TF r = 0; ///<
            TF p = 0; ///<
            TF z = 0; ///<
            TF q = 0; ///<
        };
    };


    test_with_Pc<Pc>();
}
