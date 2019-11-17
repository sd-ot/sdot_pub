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
        // P( area );
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

template<class CP,class Dirac>
typename CP::TF der0_cp( CP &cp, Dirac &d0 ) {
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
            der_0 += b_der;
        }
    }, d0.weight );

    return der_0;
}

template<class CP,class Dirac>
typename CP::TF mean_ext_mass( CP &cp, Dirac &d0 ) {
    using TF = typename CP::TF;

    // derivative
    TF res = 0, den = 0;
    cp.for_each_boundary_measure( [&]( TF /*boundary_measure*/, Dirac *d1 ) {
        if ( d1 && d1 != &d0 ) {
            res += d1->area;
            den += 1;
        }
    }, d0.weight );

    return res / den;
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

template<class TF>
TF min_interp( const std::vector<TF> &xs, const std::vector<TF> &ys ) {
    // P( x ) = y0 * ( x - x1 ) * ( x - x2 ) / ( ( x0 - x2 ) * ( x0 - x1 )  ) +
    //          y1 * ( x - x0 ) * ( x - x2 ) / ( ( x1 - x0 ) * ( x1 - x2 )  ) +
    //          y2 * ( x - x0 ) * ( x - x1 ) / ( ( x2 - x0 ) * ( x2 - x1 )  )
    // Q = P * ( x0 - x2 ) * ( x0 - x1 ) * ( x1 - x2 )

    // Q( x ) = y0 * ( x - x1 ) * ( x - x2 ) * ( x1 - x2 ) +
    //          y1 * ( x - x0 ) * ( x - x2 ) * ( x2 - x0 ) +
    //          y2 * ( x - x0 ) * ( x - x1 ) * ( x0 - x1 )

    // x² => y0 * ( x1 - x2 ) +
    //       y1 * ( x2 - x0 ) +
    //       y2 * ( x0 - x1 )
    // x  => y0 * ( x1 + x2 ) * ( x1 - x2 ) +
    //       y1 * ( x2 + x0 ) * ( x2 - x0 ) +
    //       y2 * ( x0 + x1 ) * ( x0 - x1 )

    // mul by the prods
    TF num = ys[ 0 ] * ( xs[ 1 ] - xs[ 2 ] ) * ( xs[ 1 ] + xs[ 2 ] ) +
             ys[ 1 ] * ( xs[ 2 ] - xs[ 0 ] ) * ( xs[ 2 ] + xs[ 0 ] ) +
             ys[ 2 ] * ( xs[ 0 ] - xs[ 1 ] ) * ( xs[ 0 ] + xs[ 1 ] ) ;
    TF den = ys[ 0 ] * ( xs[ 1 ] - xs[ 2 ] ) +
             ys[ 1 ] * ( xs[ 2 ] - xs[ 0 ] ) +
             ys[ 2 ] * ( xs[ 0 ] - xs[ 1 ] ) ;
    return 0.5 * num / den;
}

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
    using std::pow;
    using std::max;
    using std::min;

    // load
    std::size_t nb_diracs = 20;
    std::vector<Dirac> diracs( nb_diracs );
    std::vector<TF> positions( nb_diracs + 1, 0 );
    for( std::size_t n = 0; n < nb_diracs; ++n ) {
        //        for( std::size_t d = 0; d < dim; ++d )
        //            diracs[ n ].pos[ d ] = ( 0.2 + 0.6 * rand() / RAND_MAX ) * ( d == 0 );
        //        diracs[ n ].pos[ 0 ] = 0.2 + 0.6 * n / nb_diracs + 0.1 * rand() / RAND_MAX;
        //        diracs[ n ].pos[ 0 ] = 0.2 + 0.6 * rand() / RAND_MAX;
        diracs[ n ].pos[ 0 ] = n / ( nb_diracs - 1.0 );
        diracs[ n ].pos[ 1 ] = 0;
        diracs[ n ].weight = 1e-3 * rand() / RAND_MAX; // 0 * sin( diracs[ n ].pos.x ) * sin( diracs[ n ].pos.y );
        diracs[ n ].index = n;

        positions[ n + 1 ] = diracs[ n ].pos[ 0 ];
    }

    // grid
    Grid grid( 128 );
    grid.construct( diracs.data(), diracs.size() );

    // solve init => first residual,
    const CP ic( typename CP::Box{ { 0, 0 }, { 1, 1 } }, nullptr );
    const TF target_mass = TF( 1 ) / nb_diracs;

    //
    for( std::size_t num_iter = 0; num_iter < 15; ++num_iter ) {
        // dxn => search dir
        MtVal<TF> err;
        grid.for_each_laguerre_cell( [&]( CP &cp, Dirac &d0, int num_thread ) {
            TF dm = target_mass - cp.integral();
            err[ num_thread ] += pow( dm, 2 );
            d0.old_weight = d0.weight;
            d0.dxo = d0.dxn;
            d0.dxn = dm;

            if ( num_iter == 0 ) {
                d0.sn = dm;
                d0.so = dm;
            }
        }, ic );


        display( grid, va_string( "vtk/pd_{}.vtk", num_iter ), "vtk/grid.vtk" );

        // sn, from a correction of dxn
        if ( num_iter ) {
            MtVal<TF> dpr_num, dpr_den;
            grid.for_each_dirac( [&]( Dirac &d0, int num_thread ) {
                // FR
                dpr_num[ num_thread ] += d0.dxn * d0.dxn;
                dpr_den[ num_thread ] += d0.dxo * d0.dxo;
                // DY
                // dpr_num[ num_thread ] += d0.dxn * d0.dxn;
                // dpr_den[ num_thread ] += d0.so * ( d0.dxn - d0.dxo );
            } );

            // TF beta = max( TF( 0 ), dpr_num.sum() / dpr_den.sum() );
            TF beta = dpr_num.sum() / dpr_den.sum();
            grid.for_each_dirac( [&]( Dirac &d0, int /*num_thread*/ ) {
                d0.so = d0.sn;
                d0.sn = d0.dxn + beta * d0.so;
            } );
        }

        // try several alphas
        std::vector<TF> alphas = { 0 };
        std::vector<TF> errors = { err.sum() };
        for( TF alpha = 2; alphas.size() < 3; alpha *= 0.5 ) {
            grid.for_each_dirac( [&]( Dirac &d0, int /*num_thread*/ ) {
                d0.weight = d0.old_weight + alpha * d0.sn;
            }, { .mod_weights = true } );

            MtVal<TF> err, nb_bad_cells;
            //            grid.for_each_laguerre_cell( [&]( CP &cp, Dirac &d0, int /*num_thread*/ ) {
            //                d0.area = cp.integral();
            //            }, ic );
            grid.for_each_laguerre_cell( [&]( CP &cp, Dirac &/*d0*/, int num_thread ) {
                 // TF mass = d0.area, dm = mean_ext_mass( cp, d0 ) - d0.area;
                 TF mass = cp.integral(), dm = target_mass - mass;
                nb_bad_cells[ num_thread ] += mass == 0;
                err[ num_thread ] += pow( dm, 2 );
            }, ic );

            if ( nb_bad_cells.sum() == 0 ) {
                alphas.push_back( alpha );
                errors.push_back( err.sum() );
            }
        }

        TF best_alpha = min( alphas[ 1 ], max( TF( 1e-5 * alphas[ 1 ] ), min_interp( alphas, errors ) ) );
        P( alphas[ 1 ], best_alpha, err.sum() );
        grid.for_each_dirac( [&]( Dirac &d0, int /*num_thread*/ ) {
            d0.weight = d0.old_weight + best_alpha * d0.sn;
        }, { .mod_weights = true } );
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
            static std::vector<std::string> names() { return { "weight", "index", "dxn" }; }
            std::vector<TF> values() const { return { weight, TF( index ), dxn }; }

            //
            TF old_weight;
            TF weight;
            TI index;
            Pt pos;

            TF area;

            TF dxn;
            TF dxo;

            TF sn;
            TF so;
        };
    };

    test_with_Pc<Pc>();
}
