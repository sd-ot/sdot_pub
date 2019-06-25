#include "Geometry/Internal/FrontZgrid.h"
#include "Geometry/Internal/ZCoords.h"
#include "Support/StaticRange.h"
#include "Support/RadixSort.h"
#include "Support/Span.h"
#include "ZGrid.h"
#include <cmath>
#include <set>

namespace sdot {

template<class Pc>
ZGrid<Pc>::ZGrid( std::size_t max_diracs_per_cell ) : max_diracs_per_cell( max_diracs_per_cell ) {
}

template<class Pc>
void ZGrid<Pc>::update( const Pt *positions, const TF *weights, std::size_t nb_diracs, bool positions_have_changed, bool weights_have_changed, bool ball_cut ) {
    if ( ball_cut )
        update( positions, weights, nb_diracs, positions_have_changed, weights_have_changed, N<1>() );
    else
        update( positions, weights, nb_diracs, positions_have_changed, weights_have_changed, N<0>() );
}

template<class Pc> template<int bc>
void ZGrid<Pc>::update( const Pt *positions, const TF *weights, std::size_t nb_diracs, bool positions_have_changed, bool weights_have_changed, N<bc> ) {
    if ( positions_have_changed || weights_have_changed ) {
        update_the_limits( positions, weights, nb_diracs );
        fill_the_grid    ( positions, weights, nb_diracs );
    }
}

//template<class Pc> template<int ball_cut>
//typename ZGrid<Pc>::TF ZGrid<Pc>::min_w_to_cut( const CP &lc, const Pt c0, const TF w0, const Cell &cr_cell, const Pt *positions, const TF *weights, N<ball_cut> ) {
//    using std::sqrt;
//    using std::pow;
//    using std::max;
//    using std::min;

//    //
//    const Pt cc = cr_cell.pos + TF( 0.5 * cr_cell.size );
//    const TF sc = sqrt( TF( 0.5 ) ) * cr_cell.size;
//    if ( ball_cut )
//        return pow( max( norm_2( cc - c0 ) - sc - sqrt( w0 ), TF( 0 ) ), 2 );

//    //
//    TF res = std::numeric_limits<TF>::max();
//    if ( dim == 2 ) {
//        for( TI num_lc_point = 0; num_lc_point < lc.nb_points; ++num_lc_point ) {
//            TF nc = pow( lc.points[ 0 ][ num_lc_point ] - cc[ 0 ], 2 ) + pow( lc.points[ 1 ][ num_lc_point ] - cc[ 1 ], 2 );
//            TF n0 = pow( lc.points[ 0 ][ num_lc_point ] - c0[ 0 ], 2 ) + pow( lc.points[ 1 ][ num_lc_point ] - c0[ 1 ], 2 );
//            res = min( res, pow( max( sqrt( nc ) - sc, TF( 0 ) ), 2 ) - n0 );
//        }
//    } else {
//        for( TI num_lc_point = 0; num_lc_point < lc.nb_points; ++num_lc_point ) {
//            Pt p = lc.point( num_lc_point );
//            res = min( res, pow( max( norm_2( p - cc ) - sc, TF( 0 ) ), 2 ) - norm_2_p2( p - c0 ) );
//        }
//    }
//    return res + w0;
//}

////template<class Pc>
////bool ZGrid<Pc>::may_cut( const CP &lc, TI i0, const Grid &cr_grid, const Cell &cr_cell, const Pt *positions, const TF *weights ) {
////    using std::sqrt;
////    using std::max;
////    using std::min;
////    using std::pow;
////    using std::abs;

////    auto c0 = positions[ i0 ];
////    auto w0 = weights[ i0 ];

////    //
////    if ( ball_cut ) {
////        TF md = 0; // min dist pow 2
////        for( size_t d = 0; d < dim; ++d ) {
////            TF o = c0[ d ] - cr_cell.pos[ d ];
////            if ( o > 0 )
////                o = max( TF( 0 ), o - cr_cell.size );
////            md += pow( o, 2 );
////        }

////        return md < pow( sqrt( cr_grid.max_weight ) + sqrt( w0 ), 2 );
////    }

////    //
////    if ( full_may_cut_test ) {
////        const Pt A{ cr_cell.pos.x               , cr_cell.pos.y                };
////        const Pt B{ cr_cell.pos.x + cr_cell.size, cr_cell.pos.y                };
////        const Pt C{ cr_cell.pos.x               , cr_cell.pos.y + cr_cell.size };
////        const Pt D{ cr_cell.pos.x + cr_cell.size, cr_cell.pos.y + cr_cell.size };

////        // p = center. l = first point of the segment, d = segment direction. s = size of the segment
////        auto inter_disc_segment = [&]( Pt p, TF r2, Pt l, int d, TF min_s, TF max_s ) {
////            Pt q = p - l;
////            TF c = norm_2_p2( q ) - r2;
////            TF e = q[ d ] * q[ d ] - c;
////            if ( e <= 0 )
////                return false;
////            e = sqrt( e );

////            TF s0 = q[ d ] + e;
////            TF s1 = q[ d ] - e;
////            return ( s0 >= min_s && s0 <= max_s ) || ( s1 >= min_s && s1 <= max_s );
////        };

////        // l = first point of the segment. d = direction of the segment. s = size of the segment
////        auto test_one_line = [&]( Pt p, TF r2, Pt l0, Pt l1, int d, TF s, TF d_eps ) {
////            return norm_2_p2( l0 - p ) < r2 + d_eps * d_eps ||
////                    norm_2_p2( l1 - p ) < r2 + d_eps * d_eps ||
////                    inter_disc_segment( p, r2, l0, d, 0 - d_eps, s + d_eps );
////        };

////        // l = common point of the segments. sx = signed sizes of the segments for for the x direction
////        auto test_two_lines = [&]( Pt p, TF r2, Pt lm, Pt lx, Pt ly, TF min_x, TF max_x, TF min_y, TF max_y, TF d_eps ) {
////            return norm_2_p2( lx - p ) < r2 + d_eps * d_eps ||
////                    norm_2_p2( ly - p ) < r2 + d_eps * d_eps ||
////                    inter_disc_segment( p, r2, lm, 0, min_x, max_x ) ||
////                    inter_disc_segment( p, r2, lm, 1, min_y, max_y ) ;
////        };

////        //
////        const TF d_eps = 10 * std::numeric_limits<TF>::epsilon() * cr_cell.size;
////        for( std::size_t num_lc_point = 0; num_lc_point < lc.nb_points; ++num_lc_point ) {
////            Pt p = lc.point( num_lc_point );
////            TF r2 = norm_2_p2( p - c0 ) + cr_grid.max_weight - w0;
////            if ( r2 <= 0 )
////                continue;

////            if ( p.x < A.x - d_eps ) {
////                if ( p.y < A.y ) {
////                    if ( test_two_lines( p, r2, A, B, C, - d_eps, cr_cell.size + d_eps, - d_eps, cr_cell.size + d_eps, d_eps ) )
////                        return true;
////                } else if ( p.y < C.y ) {
////                    if ( test_one_line( p, r2, A, C, 1, cr_cell.size, d_eps ) )
////                        return true;
////                } else {
////                    if ( test_two_lines( p, r2, C, D, A, - d_eps, cr_cell.size + d_eps, - cr_cell.size - d_eps, d_eps, d_eps ) )
////                        return true;
////                }
////            } else if ( p.x < B.x + d_eps ) {
////                if ( p.y < A.y - d_eps ) {
////                    if ( test_one_line( p, r2, A, B, 0, cr_cell.size, d_eps ) )
////                        return true;
////                } else if ( p.y < C.y + d_eps ) {
////                    return true;
////                } else {
////                    if ( test_one_line( p, r2, C, D, 0, cr_cell.size, d_eps ) )
////                        return true;
////                }
////            } else {
////                if ( p.y < A.y ) {
////                    if ( test_two_lines( p, r2, B, A, D, - cr_cell.size - d_eps, d_eps, - d_eps, cr_cell.size + d_eps, d_eps ) )
////                        return true;
////                } else if ( p.y < C.y ) {
////                    if ( test_one_line( p, r2, B, D, 1, cr_cell.size, d_eps ) )
////                        return true;
////                } else {
////                    if ( test_two_lines( p, r2, D, C, B, - cr_cell.size - d_eps, d_eps, - cr_cell.size - d_eps, d_eps, d_eps ) )
////                        return true;
////                }
////            }
////        }
////    } else {
////        const Pt C = cr_cell.pos + TF( 0.5 ) * cr_cell.size;
////        const TF s = sqrt( TF( 0.5 ) ) * cr_cell.size;
////        for( TI num_lc_point = 0; num_lc_point < lc.nb_points; ++num_lc_point ) {
////            Pt p = lc.point( num_lc_point );
////            TF r2 = norm_2_p2( p - c0 ) + cr_grid.max_weight - w0;
////            if ( r2 > 0 && norm_2( p - C ) < sqrt( r2 ) + s )
////                return true;
////        }
////    }

////    return false;
////}

template<class Pc>
int ZGrid<Pc>::for_each_laguerre_cell( const std::function<void( CP &, TI num, int num_thread )> &cb, const CP &starting_lc, const Pt *positions, const TF *weights, TI nb_diracs, bool stop_if_void_lc, bool ball_cut ) {
    return ball_cut ?
        for_each_laguerre_cell( cb, starting_lc, positions, weights, nb_diracs, stop_if_void_lc, N<1>() ) :
        for_each_laguerre_cell( cb, starting_lc, positions, weights, nb_diracs, stop_if_void_lc, N<0>() ) ;
}

template<class Pc> template<int ball_cut>
int ZGrid<Pc>::for_each_laguerre_cell( const std::function<void( CP &, TI num, int num_thread )> &cb, const CP &starting_lc, const Pt *positions, const TF *weights, TI /*nb_diracs*/, bool stop_if_void_lc, N<ball_cut> ) {
    using Front = FrontZgrid<ZGrid>;
    using std::sqrt;

    auto cut = [&]( Pt c0, TF w0, TI i1 ) -> typename CP::Cut {
        Pt V = positions[ i1 ] - c0;
        TF n = norm_2_p2( V );
        TF x = TF( 0.5 ) + TF( 0.5 ) * ( w0 - weights[ i1 ] ) / n;
        return { V, dot( c0 + x * V, V ), i1 };
    };

    // vectors for stuff that will be reused inside the execution threads
    int nb_threads = thread_pool.nb_threads(), nb_jobs = 4 * nb_threads;
    std::vector<std::vector<TI>> visited( nb_threads );
    std::vector<TI> op_counts( nb_threads, 0 );

    for( int num_thread = 0; num_thread < nb_threads; ++num_thread )
        visited[ num_thread ].resize( grid.cells.size(), op_counts[ num_thread ] );

    // for each item
    int err = 0;
    thread_pool.execute( nb_jobs, [&]( std::size_t num_job, int num_thread ) {
        Front front( op_counts[ num_thread ], visited[ num_thread ] );
        typename CP::Cut cuts[ 32 ];
        CP lc;

        TI beg_cell = ( num_job + 0 ) * ( grid.cells.size() - 1 ) / nb_jobs;
        TI end_cell = ( num_job + 1 ) * ( grid.cells.size() - 1 ) / nb_jobs;
        for( TI num_cell = beg_cell; num_cell < end_cell && err == 0; ++num_cell ) {
            const Cell &cell = grid.cells[ num_cell + 0 ];
            const Cell &dell = grid.cells[ num_cell + 1 ];
            for( TI num_dirac : Span<TI>{ grid.dpc_values.data() + cell.dpc_offset, grid.dpc_values.data() + grid.cells[ num_cell + 1 ].dpc_offset } ) {
                const Pt c0 = positions[ num_dirac ];
                const TF w0 = weights[ num_dirac ];

                // start of lc: cut with nodes in the same cell
                lc = starting_lc;

                //
                TI nb_cuts = 0;
                for( TI num_cr_dirac : Span<TI>{ grid.dpc_values.data(), cell.dpc_offset, dell.dpc_offset } )
                    if ( num_cr_dirac != num_dirac )
                        cuts[ nb_cuts++ ] = cut( c0, w0, num_cr_dirac );
                lc.plane_cut( cuts, nb_cuts );

                // front
                front.init( num_cell, positions[ num_dirac ], weights[ num_dirac ] );

                // => neighbors in the same grid
                for( TI num_ng_cell : Span<TI>{ grid.ng_indices.data() + grid.ng_offsets[ num_cell + 0 ], grid.ng_indices.data() + grid.ng_offsets[ num_cell + 1 ] } )
                    front.push_without_check( num_ng_cell, grid );

                // neighbors
                while( ! front.empty() ) {
                    typename Front::Item cr = front.pop();
                    const Cell &cr_cell = grid.cells[ cr.num_cell ];

                    //                    // if no cut is possible, we don't go further.
                    //                    TF min_w = min_w_to_cut( lc, c0, w0, cr_cell, positions, weights, N<ball_cut>() );
                    //                    if ( grid.max_weight <= min_w )
                    //                        continue;

                    // if we have diracs in cr that may cut, try them
                    // if ( cr_cell.max_weight > min_w )
                    TI nb_cuts = 0;
                    for( TI num_cr_dirac : Span<TI>{ grid.dpc_values.data() + cr_cell.dpc_offset, grid.dpc_values.data() + grid.cells[ cr.num_cell + 1 ].dpc_offset } )
                        //                        if ( weights[ num_cr_dirac ] > min_w )
                        cuts[ nb_cuts++ ] = cut( c0, w0, num_cr_dirac );
                    lc.plane_cut( cuts, nb_cuts );

                    // update the front
                    for( TI num_ng_cell : Span<TI>{ grid.ng_indices.data() + grid.ng_offsets[ cr.num_cell + 0 ], grid.ng_indices.data() + grid.ng_offsets[ cr.num_cell + 1 ] } )
                        front.push( num_ng_cell, grid );
                }

                //
                //                if ( ball_cut )
                //                    lc.ball_cut( positions[ num_dirac ], sqrt( weights[ num_dirac ] ), num_dirac );
                //                else
                //                    lc.sphere_center = positions[ num_dirac ];

                //
                if ( stop_if_void_lc && lc.empty() ) {
                    err = 1;
                    break;
                }

                //
                cb( lc, num_dirac, num_thread );
            }
        }
    } );

    return err;
}

//template<class Pc>
//bool ZGrid<Pc>::check_sanity( const Pt *positions ) const {
//    if ( grids.size() == 0 )
//        return false;

//    // check diracs appear only once
//    std::vector<bool> c( grids[ 0 ].cell_index_vs_dirac_number.size(), false );
//    for( const Grid &grid : grids ) {
//        ASSERT( grid.cell_index_vs_dirac_number.size() == grids[ 0 ].cell_index_vs_dirac_number.size(), "" );
//        for( std::size_t num_cell = 0; num_cell < grid.cells.size() - 1; ++num_cell ) {
//            for( TI num_dirac : Span<TI>{ grid.dpc_values.data() + grid.cells[ num_cell + 0 ].dpc_offset, grid.dpc_values.data() + grid.cells[ num_cell + 1 ].dpc_offset } ) {
//                ASSERT( c[ num_dirac ] == false, "" );
//                c[ num_dirac ] = true;
//            }
//        }
//    }

//    // check cell_index_vs_dirac_number: diracs must be inside
//    for( std::size_t num_dirac = 0; num_dirac < grids[ 0 ].cell_index_vs_dirac_number.size(); ++num_dirac ) {
//        for( const Grid &grid : grids ) {
//            const Cell &cell = grid.cells[ grid.cell_index_vs_dirac_number[ num_dirac ] ];
//            for( std::size_t d = 0; d < dim; ++d ) {
//                ASSERT( positions[ num_dirac ][ d ] <= cell.pos[ d ] + cell.size, "" );
//                ASSERT( positions[ num_dirac ][ d ] >= cell.pos[ d ], "" );
//            }
//        }
//    }

//    // check neighbors
//    for( const Grid &grid : grids ) {
//        std::vector<std::vector<TI>> neighbors( grid.cells.size() - 1 );
//        auto ae = []( TF a, TF b ) {
//             using std::abs;
//             return abs( a - b ) < 1e-6;
//        };

//        auto touching = [&]( const Cell &c0, const Cell &c1 ) {
//            using std::min;
//            using std::max;
//            return ( ae( c0.pos.x + c0.size, c1.pos.x ) && min( c0.pos.y + c0.size, c1.pos.y + c1.size ) - max( c0.pos.y, c1.pos.y ) > 1e-6 ) || // left
//                   ( ae( c0.pos.x, c1.pos.x + c1.size ) && min( c0.pos.y + c0.size, c1.pos.y + c1.size ) - max( c0.pos.y, c1.pos.y ) > 1e-6 ) || // right
//                   ( min( c0.pos.x + c0.size, c1.pos.x + c1.size ) - max( c0.pos.x, c1.pos.x ) > 1e-6 && ae( c0.pos.y + c0.size, c1.pos.y ) ) || // up
//                   ( min( c0.pos.x + c0.size, c1.pos.x + c1.size ) - max( c0.pos.x, c1.pos.x ) > 1e-6 && ae( c0.pos.y, c1.pos.y + c1.size ) ) ;  // bottom
//        };

//        for( std::size_t num_cell_0 = 0; num_cell_0 < grid.cells.size() - 1; ++num_cell_0 )
//            for( std::size_t num_cell_1 = 0; num_cell_1 < grid.cells.size() - 1; ++num_cell_1 )
//                if ( num_cell_0 != num_cell_1 && touching( grid.cells[ num_cell_0 ], grid.cells[ num_cell_1 ] ) )
//                    neighbors[ num_cell_0 ].push_back( num_cell_1 );

//        for( std::size_t num_cell = 0; num_cell < grid.cells.size() - 1; ++num_cell )
//            if ( grid.ng_offsets[ num_cell + 1 ] - grid.ng_offsets[ num_cell + 0 ] != neighbors[ num_cell ].size() ) {
//                //                P( num_cell );
//                //                P( neighbors[ num_cell ] );
//                //                for( TI v : Span<TI>( grid.ng_indices.data() + grid.ng_offsets[ num_cell + 0 ], grid.ng_indices.data() + grid.ng_offsets[ num_cell + 1 ] ) )
//                //                    P( v );
//                ASSERT( neighbors[ num_cell ].size() == grid.ng_offsets[ num_cell + 1 ] - grid.ng_offsets[ num_cell + 0 ], "" );
//            }
//    }

//    return true;
//}

template<class Pc>
void ZGrid<Pc>::update_the_limits( const Pt *positions, const TF *weights, std::size_t nb_diracs ) {
    using std::min;
    using std::max;

    // min/max
    for( std::size_t d = 0; d < dim; ++d ) {
        min_point[ d ] = + std::numeric_limits<TF>::max();
        max_point[ d ] = - std::numeric_limits<TF>::max();
    }
    min_weight = + std::numeric_limits<TF>::max();
    max_weight = - std::numeric_limits<TF>::max();

    for( std::size_t num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
        for( std::size_t d = 0; d < dim; ++d ) {
            min_point[ d ] = min( min_point[ d ], positions[ num_dirac ][ d ] );
            max_point[ d ] = max( max_point[ d ], positions[ num_dirac ][ d ] );
        }
        min_weight = min( min_weight, weights[ num_dirac ] );
        max_weight = max( max_weight, weights[ num_dirac ] );
    }

    // grid size
    grid_length = 0;
    for( std::size_t d = 0; d < dim; ++d )
        grid_length = max( grid_length, max_point[ d ] - min_point[ d ] );
    grid_length *= 1 + std::numeric_limits<TF>::epsilon();

    step_length = grid_length / ( TZ( 1 ) << nb_bits_per_axis );
}


template<class Pc>
void ZGrid<Pc>::update_neighbors() {
    // make a list of requests to get the neighbors
    znodes.resize( 0 );
    znodes.reserve( 2 * dim * zcells.size() );
    for( TI num_cell = 0; num_cell < zcells.size() - 1; ++num_cell ) {
        constexpr TZ f00 = ~ ( ( TZ( 1 ) << dim * nb_bits_per_axis ) - 1 );
        StaticRange<dim>::for_each( [&]( auto d ) {
            TZ z0 = zcells[ num_cell + 0 ].zcoords;
            TZ z1 = zcells[ num_cell + 1 ].zcoords;
            TZ nz = ng_zcoord( z0, z1 - z0, d );
            if ( ( nz & f00 ) == 0 ) // test for overflow
                znodes.push_back( { nz, num_cell } );
        } );
    }

    znodes.reserve( 2 * znodes.size() );
    ZNode *sorted_znodes = radix_sort( znodes.data() + znodes.size(), znodes.data(), znodes.size(), N<sizeof_zcoords>(), rs_tmps );

    // helper function to get the neighbors for each node
    auto for_each_ng = [&]( auto cb ) {
        for( TI i = 0, j = 0; i < znodes.size(); ++i ) {
            // find first node with zcoords > sorted_znodes[ i ].zcoords
            while ( zcells[ j ].zcoords <= sorted_znodes[ i ].zcoords )
                ++j;

            // first node
            TI index_node = sorted_znodes[ i ].index;
            TI index_nbor = j - 1;
            cb( index_node, index_nbor );

            // next touching ones
            TZ off = zcells[ index_node + 1 ].zcoords - zcells[ index_node ].zcoords;
            TZ lim = zcells[ index_nbor ].zcoords + off;
            if ( zcells[ index_nbor + 1 ].zcoords < lim ) {
                // find direction to ng
                StaticRange<dim>::for_each_cont( [&]( auto dir ) {
                    using Zooa = typename ZCoords<TZ,dim,nb_bits_per_axis>::template _ZcoordsOnesOnAxis<dir.val>;
                    if ( ( zcells[ index_node ].zcoords & Zooa::value ) != ( zcells[ index_nbor ].zcoords & Zooa::value ) ) {
                        TZ dv = zcells[ index_nbor++ ].zcoords & Zooa::value;
                        do {
                            if ( ( zcells[ index_nbor ].zcoords & Zooa::value ) == dv )
                                cb( index_node, index_nbor );
                        } while ( zcells[ ++index_nbor ].zcoords < lim );
                        return false;
                    }
                    return true;
                } );
            }
        }
    };

    //
    grid.ng_offsets.resize( zcells.size() );
    for( TI i = 0; i < zcells.size(); ++i )
        grid.ng_offsets[ i ] = 0;

    // get count
    for_each_ng( [&]( std::size_t index_node, std::size_t index_nbor ) {
        ++grid.ng_offsets[ index_node ];
        ++grid.ng_offsets[ index_nbor ];
    } );

    // suffix scan
    for( TI i = 0, acc = 0; i < zcells.size(); ++i ) {
        TI v = acc;
        acc += grid.ng_offsets[ i ];
        grid.ng_offsets[ i ] = v;
    }

    // get indices
    grid.ng_indices.resize( grid.ng_offsets.back() );
    for_each_ng( [&]( std::size_t index_node, std::size_t index_nbor ) {
        grid.ng_indices[ grid.ng_offsets[ index_node ]++ ] = index_nbor;
        grid.ng_indices[ grid.ng_offsets[ index_nbor ]++ ] = index_node;
    } );

    // shift ng_offsets (to get the suffix scan again)
    if ( grid.ng_offsets.size() ) {
        for( TI i = grid.ng_offsets.size(); --i; )
            grid.ng_offsets[ i ] = grid.ng_offsets[ i - 1 ];
        grid.ng_offsets[ 0 ] = 0;
    }
}

template<class Pc>
void ZGrid<Pc>::fill_grid_using_zcoords( const Pt *positions, const TF */*weights*/, std::size_t nb_diracs ) {
    using std::round;
    using std::ceil;
    using std::pow;
    using std::min;
    using std::max;

    // get zcoords for each dirac
    znodes.clear();
    znodes.reserve( 2 * nb_diracs );
    for( TI index = 0; index < nb_diracs; ++index )
        znodes.push_back( { zcoords_for( positions[ index ] ), index } );

    // prepare cell_index_vs_dirac_number => we will set the values for the diracs in this grid
    grid.cell_index_vs_dirac_number.resize( nb_diracs, 666000 );

    // sorting w.r.t. zcoords
    znodes.reserve( 2 * znodes.size() );
    ZNode *sorted_znodes = radix_sort( znodes.data() + znodes.size(), znodes.data(), znodes.size(), N<sizeof_zcoords>(), rs_tmps );

    // fill `cells` with zcoords
    int level = 0;
    TZ prev_z = 0;
    zcells.resize( 0 );
    zcells.reserve( znodes.size() );
    grid.dpc_values.resize( 0 );
    grid.dpc_values.reserve( znodes.size() );
    for( TI index = max_diracs_per_cell; ; ) {
        if ( index >= znodes.size() ) {
            while ( prev_z < ( TZ( 1 ) << dim * nb_bits_per_axis ) ) {
                for( ; ; ++level ) {
                    TZ m = TZ( 1 ) << dim * ( level + 1 );
                    if ( level == nb_bits_per_axis || prev_z & ( m - 1 ) ) {
                        TZ new_prev_z = prev_z + ( TZ( 1 ) << dim * level );

                        ZNode cell;
                        cell.zcoords = prev_z;

                        cell.index = grid.dpc_values.size();
                        for( TI n = index - max_diracs_per_cell; n < znodes.size(); ++n ) {
                            if ( sorted_znodes[ n ].zcoords >= prev_z && sorted_znodes[ n ].zcoords < new_prev_z ) {
                                grid.cell_index_vs_dirac_number[ sorted_znodes[ n ].index ] = zcells.size();
                                grid.dpc_values.push_back( sorted_znodes[ n ].index );
                                ++index;
                            }
                        }

                        zcells.push_back( cell );
                        prev_z = new_prev_z;
                        break;
                    }
                }
            }
            break;
        }

        // level too high ?
        for( ; ; --level ) {
            TZ m = TZ( 1 ) << dim * ( level + 1 );
            if ( sorted_znodes[ index ].zcoords >= prev_z + m )
                break;
            ASSERT( level, "Seems not possible to have $max_diracs_per_cell considering the discretisation (some points are too close)" );
        }

        // look for a level before the one that will take the $max_diracs_per_cell next points or that will lead to an illegal cell
        for( ; ; ++level ) {
            TZ m = TZ( 1 ) << dim * ( level + 1 );
            if ( sorted_znodes[ index ].zcoords < prev_z + m || ( prev_z & ( m - 1 ) ) ) {
                TZ new_prev_z = prev_z + ( TZ( 1 ) << dim * level );

                ZNode zcell;
                zcell.zcoords = prev_z;

                zcell.index = grid.dpc_values.size();
                for( TI n = index - max_diracs_per_cell, l = index; n < l; ++n ) {
                    if ( sorted_znodes[ n ].zcoords >= prev_z && sorted_znodes[ n ].zcoords < new_prev_z ) {
                        grid.cell_index_vs_dirac_number[ sorted_znodes[ n ].index ] = zcells.size();
                        grid.dpc_values.push_back( sorted_znodes[ n ].index );
                        ++index;
                    }
                }

                zcells.push_back( zcell );
                prev_z = new_prev_z;
                break;
            }
        }
    }

    // add an ending cell
    ZNode zcell;
    zcell.index = grid.dpc_values.size();
    zcell.zcoords = TZ( 1 ) << dim * nb_bits_per_axis;
    zcells.push_back( zcell );
}

template<class Pc>
void ZGrid<Pc>::fill_the_grid( const Pt *positions, const TF *weights, std::size_t nb_diracs ) {
    static_assert( sizeof( TZ ) >= sizeof_zcoords, "zcoords types is not large enough" );

    // set grid content
    fill_grid_using_zcoords( positions, weights, nb_diracs );
    update_neighbors       ();
    repl_zcoords_by_ccoords( weights );
}

template<class Pc> template<class C>
typename ZGrid<Pc>::TZ ZGrid<Pc>::zcoords_for( const C &pos ) {
    std::array<TZ,dim> c;
    for( int d = 0; d < dim; ++d )
        c[ d ] = TZ( TF( TZ( 1 ) << nb_bits_per_axis ) * ( pos[ d ] - min_point[ d ] ) / grid_length );

    TZ res = 0;
    switch ( dim ) {
    case 1:
        res = c[ 0 ];
        break;
    case 2:
        for( int o = 0; o < nb_bits_per_axis; o += 8 )
            res |= TZ( morton_256_2D_x[ ( c[ 0 ] >> o ) & 0xFF ] |
                       morton_256_2D_y[ ( c[ 1 ] >> o ) & 0xFF ] ) << dim *  o;
        break;
    case 3:
        for( int o = 0; o < nb_bits_per_axis; o += 8 )
            res |= TZ( morton_256_3D_x[ ( c[ 0 ] >> o ) & 0xFF ] |
                       morton_256_3D_y[ ( c[ 1 ] >> o ) & 0xFF ] |
                       morton_256_3D_z[ ( c[ 2 ] >> o ) & 0xFF ] ) << dim *  o;
        break;
    default:
        TODO;
    }

    return res;
}

template<class Pc>
void ZGrid<Pc>::display_tikz( std::ostream &os ) const {
    for( TI num_cell = 0; num_cell < grid.cells.size() - 1; ++num_cell ) {
        Pt p;
        for( int d = 0; d < dim; ++d )
            p[ d ] = grid.cells[ num_cell ].pos[ d ];

        TF a = 0, b = grid.cells[ num_cell ].size;
        switch ( dim ) {
        case 2:
            os << "\\draw ";
            os << "(" << p[ 0 ] + a << "," << p[ 1 ] + a << ") -- ";
            os << "(" << p[ 0 ] + b << "," << p[ 1 ] + a << ") -- ";
            os << "(" << p[ 0 ] + b << "," << p[ 1 ] + b << ") -- ";
            os << "(" << p[ 0 ] + a << "," << p[ 1 ] + b << ") -- ";
            os << "(" << p[ 0 ] + a << "," << p[ 1 ] + a << ") ;\n";
            break;
        case 3:
            TODO;
            break;
        default:
            TODO;
        }
    }
}

template<class Pc>
void ZGrid<Pc>::display( VtkOutput &vtk_output ) const {
    for( TI num_cell = 0; num_cell < grid.cells.size() - 1; ++num_cell ) {
        Pt p;
        for( int d = 0; d < dim; ++d )
            p[ d ] = grid.cells[ num_cell ].pos[ d ];

        TF a = 0, b = grid.cells[ num_cell ].size;
        switch ( dim ) {
        case 2:
            vtk_output.add_lines( {
                Point2<TF>{ p[ 0 ] + a, p[ 1 ] + a },
                Point2<TF>{ p[ 0 ] + b, p[ 1 ] + a },
                Point2<TF>{ p[ 0 ] + b, p[ 1 ] + b },
                Point2<TF>{ p[ 0 ] + a, p[ 1 ] + b },
                Point2<TF>{ p[ 0 ] + a, p[ 1 ] + a },
            } );
            break;
        case 3:
            TODO;
            break;
        default:
            TODO;
        }
    }
}

template<class Pc> template<int axis>
typename ZGrid<Pc>::TZ ZGrid<Pc>::ng_zcoord( TZ zcoords, TZ off, N<axis> ) const {
    using Zzoa = typename ZCoords<TZ,dim,nb_bits_per_axis>::template _ZcoordsZerosOnAxis<axis>;
    TZ ff0 = Zzoa::value;
    TZ res = ( ( zcoords | ff0 ) + off ) & ~ ff0;
    return res | ( zcoords & ff0 );
}

template<class Pc>
void ZGrid<Pc>::repl_zcoords_by_ccoords( const TF */*weights*/ ) {
    using std::max;

    // convert zcoords to cartesian coords
    grid.cells.resize( zcells.size() );
    for( TI num_cell = 0; num_cell < grid.cells.size() - 1; ++num_cell ) {
        const ZNode &p = zcells[ num_cell + 0 ];
        const ZNode &n = zcells[ num_cell + 1 ];

        Cell &c = grid.cells[ num_cell ];
        c.size = step_length * round( pow( n.zcoords - p.zcoords, 1.0 / dim ) );
        c.zcoords = p.zcoords;
        c.dpc_offset = p.index;

        StaticRange<dim>::for_each( [&]( auto d ) {
            c.pos[ d ] = TF( 0 );
            StaticRange<nb_bits_per_axis>::for_each( [&]( auto i ) {
                c.pos[ d ] += ( p.zcoords & ( TZ( 1 ) << ( dim * i + d ) ) ) >> ( ( dim - 1 ) * i + d );
            } );
            c.pos[ d ] = min_point[ d ] + step_length * c.pos[ d ];
        } );
    }

    Cell &c = grid.cells.back();
    c.dpc_offset = zcells.back().index;
    c.zcoords = zcells.back().zcoords;
    c.size = 0;
    c.pos = max_point;
}

//template<class Pc>
//void ZGrid<Pc>::find_englobing_cousins( TI num_grid, const Pt *positions ) {
//    if ( grids.size() == 1 )
//        return;

//    // znodes for diracs of current grid
//    znodes.clear();
//    for( TI num_dirac : grids[ num_grid ].dirac_indices )
//        znodes.push_back( { zcoords_for( positions[ num_dirac ] ), num_dirac } );

//    // sort znodes
//    znodes.reserve( 2 * znodes.size() );
//    ZNode *out = radix_sort( znodes.data() + znodes.size(), znodes.data(), znodes.size(), N<sizeof_zcoords>(), rs_tmps );

//    //
//    for( std::size_t num_ot_grid = 0; num_ot_grid < grids.size(); ++num_ot_grid ) {
//        if ( num_ot_grid != num_grid ) {
//            for( std::size_t i = 0, j = 0; i < znodes.size(); ++i ) {
//                while ( grids[ num_ot_grid ].cells[ j ].zcoords <= out[ i ].zcoords )
//                    ++j;
//                grids[ num_ot_grid ].cell_index_vs_dirac_number[ out[ i ].index ] = j - 1;
//            }
//        }
//    }
//}

} // namespace sdot
