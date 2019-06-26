#include "Geometry/Internal/FrontZgrid.h"
#include "Geometry/Internal/ZCoords.h"
#include "Support/StaticRange.h"
#include "Support/RadixSort.h"
#include "Support/Stat.h"
#include "Support/Span.h"
#include "ZGrid.h"
#include <cmath>
#include <set>

namespace sdot {

template<class Pc>
ZGrid<Pc>::ZGrid( std::size_t max_diracs_per_cell ) : max_diracs_per_cell( max_diracs_per_cell ) {
}

template<class Pc> template<int flags>
void ZGrid<Pc>::update( std::array<const TF *,dim> positions, const TF *weights, std::size_t nb_diracs, N<flags>, bool positions_have_changed, bool weights_have_changed ) {
    if ( positions_have_changed || weights_have_changed ) {
        update_the_limits( positions, weights, nb_diracs );
        fill_the_grid    ( positions, weights, nb_diracs );
    }
}

template<class Pc> template<int flags>
typename ZGrid<Pc>::TF ZGrid<Pc>::min_w_to_cut( const CP &lc, const Pt &c0, const TF w0, const Cell &cr_cell, N<flags> ) {
    using std::sqrt;
    using std::pow;
    using std::max;
    using std::min;

    //
    const Pt cc = cr_cell.pos + TF( 0.5 * cr_cell.size );
    const TF sc = sqrt( TF( 0.5 ) ) * cr_cell.size;
    if ( flags & ball_cut )
        return pow( max( norm_2( cc - c0 ) - sc - sqrt( w0 ), TF( 0 ) ), 2 );

    //
    TF res = std::numeric_limits<TF>::max();
    lc.for_each_node( [&]( const auto &node ) {
        TF nc = pow( node.x - cc.x, 2 ) + pow( node.y - cc.y, 2 );
        TF n0 = pow( node.x - c0.x, 2 ) + pow( node.y - c0.y, 2 );
        res = min( res, pow( max( sqrt( nc ) - sc, TF( 0 ) ), 2 ) - n0 );
    } );

    return res + w0;
}

template<class Pc> template<int flags>
bool ZGrid<Pc>::may_cut( const CP &lc, const Pt &c0, TF w0, const Cell &cr_cell, N<flags> ) {
    using std::sqrt;
    using std::max;
    using std::min;
    using std::pow;
    using std::abs;

    //
    if ( flags & ball_cut ) {
        TF md = 0; // min dist pow 2
        for( size_t d = 0; d < dim; ++d ) {
            TF o = c0[ d ] - cr_cell.pos[ d ];
            if ( o > 0 )
                o = max( TF( 0 ), o - cr_cell.size );
            md += pow( o, 2 );
        }

        return md < pow( sqrt( max_weight ) + sqrt( w0 ), 2 );
    }

    #ifdef __AVX512F__
    if ( lc.nb_nodes() <= 8 ) {
        __m512d p_x = _mm512_load_pd( &lc.node( 0 ).x );
        __m512d p_y = _mm512_load_pd( &lc.node( 0 ).y );
        __m512d c0x = _mm512_set1_pd( c0.x );
        __m512d c0y = _mm512_set1_pd( c0.y );
        __m512d d0x = _mm512_sub_pd( p_x, c0x );
        __m512d d0y = _mm512_sub_pd( p_y, c0y );

        const TF s = TF( 0.5 ) * cr_cell.size;
        const Pt C = cr_cell.pos + s;

        // | p - c1 | => max( 0, abs( p - C ) - s )
        __m512d sp = _mm512_set1_pd( s );
        __m512d zp = _mm512_setzero_pd();
        __m512d d1x = _mm512_max_pd( zp, _mm512_sub_pd( _mm512_abs_pd( _mm512_sub_pd( p_x, _mm512_set1_pd( C.x ) ) ), sp ) );
        __m512d d1y = _mm512_max_pd( zp, _mm512_sub_pd( _mm512_abs_pd( _mm512_sub_pd( p_y, _mm512_set1_pd( C.y ) ) ), sp ) );

        // | p - c0 |^2, | p - c1 |^2
        __m512d d02 = _mm512_add_pd( _mm512_mul_pd( d0x, d0x ), _mm512_mul_pd( d0y, d0y ) );
        __m512d d12 = _mm512_add_pd( _mm512_mul_pd( d1x, d1x ), _mm512_mul_pd( d1y, d1y ) );

        if ( flags & homogeneous_weights ) {
            // | p - c1 |^2 < | p - c0 |^2
            int n = _mm512_cmp_pd_mask( d12, d02, _CMP_LT_OQ );
            if ( n & ( ( 1 << lc.nb_nodes() ) - 1 ) )
                return true;
        } else {
            // | p - c1 |^2 - | p - c0 |^2 < w1^2 - w0^2
            int n = _mm512_cmp_pd_mask( _mm512_sub_pd( d12, d02 ), _mm512_set1_pd( pow( max_weight, 2 ) - pow( w0, 2 ) ), _CMP_LT_OQ );
            if ( n & ( ( 1 << lc.nb_nodes() ) - 1 ) )
                return true;
        }
    } else {
        const Pt C = cr_cell.pos + TF( 0.5 ) * cr_cell.size;
        const TF s = sqrt( TF( 0.5 ) ) * cr_cell.size;
        for( TI num_lc_point = 0; num_lc_point < lc.nb_nodes(); ++num_lc_point ) {
            Pt p = lc.node( num_lc_point ).pos();
            TF r2 = norm_2_p2( p - c0 );
            if ( ( flags & homogeneous_weights ) == 0 )
                r2 += max_weight - w0;
            if ( pow( norm_2( p - C ) - s, 2 ) < r2 )
                return true;
        }
    }
    #else
    for( TI num_lc_point = 0; num_lc_point < lc.nb_nodes(); ++num_lc_point ) {
        const Pt C = cr_cell.pos + TF( 0.5 ) * cr_cell.size;
        const TF s = sqrt( TF( 0.5 ) ) * cr_cell.size;
        Pt p = lc.node( num_lc_point ).pos();
        TF r2 = norm_2_p2( p - c0 );
        if ( ( flags & homogeneous_weights ) == 0 )
            r2 += max_weight - w0;
        if ( pow( norm_2( p - C ) - s, 2 ) < r2 )
            return true;
    }
    #endif

    return false;
}

template<class Pc> template<int flags>
int ZGrid<Pc>::for_each_laguerre_cell( const std::function<void( CP &, TI num, int num_thread )> &cb, const CP &starting_lc, std::array<const TF *,dim> positions, const TF *weights, TI /*nb_diracs*/, N<flags>, bool stop_if_void_lc ) {
    using Front = FrontZgrid<ZGrid>;
    using std::sqrt;

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
        CP lc;

        TI beg_cell = ( num_job + 0 ) * ( grid.cells.size() - 1 ) / nb_jobs;
        TI end_cell = ( num_job + 1 ) * ( grid.cells.size() - 1 ) / nb_jobs;
        for( TI num_cell = beg_cell; num_cell < end_cell && err == 0; ++num_cell ) {
            const Cell &cell = grid.cells[ num_cell + 0 ];
            const Cell &dell = grid.cells[ num_cell + 1 ];
            for( TI i0 : Span<TI>{ grid.dpc_values.data(), cell.dpc_offset, dell.dpc_offset } ) {
                const Pt c0 = positions[ i0 ];
                const TF w0 = weights[ i0 ];

                // start of lc: cut with nodes in the same cell
                lc = starting_lc;

                //
                struct alignas(64) Cut {
                    TF dx[ 128 ];
                    TF dy[ 128 ];
                    TF ps[ 128 ];
                    CI id[ 128 ];
                };
                Cut cut;

                #ifdef __AVX512F__
                TI nb_cuts = dell.dpc_offset - cell.dpc_offset;
                TI n = 0;
                for( ; n + 8 <= nb_cuts; n += 8 ) {
                    const void *x = &positions[ 0 ].x;
                    const void *y = &positions[ 0 ].y;
                    __m512i i1 = _mm512_loadu_si512( grid.dpc_values.data() + cell.dpc_offset + n );
                    __m512i m1 = _mm512_mullo_epi64( i1, _mm512_set1_epi64( 16 ) );
                    __m512d cx = _mm512_set1_pd( c0.x );
                    __m512d cy = _mm512_set1_pd( c0.y );
                    __m512d vx = _mm512_sub_pd( _mm512_i64gather_pd( m1, x, 1 ), cx );
                    __m512d vy = _mm512_sub_pd( _mm512_i64gather_pd( m1, y, 1 ), cy );
                    __m512d v2 = _mm512_add_pd( _mm512_mul_pd( vx, vx ), _mm512_mul_pd( vy, vy ) );
                    __m512d ps = _mm512_add_pd( _mm512_add_pd( _mm512_mul_pd( cx, vx ), _mm512_mul_pd( cy, vy ) ),
                                                _mm512_set1_pd( 0.5 ) * ( flags & homogeneous_weights ? v2 : _mm512_add_pd( v2, _mm512_set1_pd( w0 ) ) - _mm512_i64gather_pd( i1, weights, 8 ) ) );
                    _mm512_store_pd( cut.dx + n, vx );
                    _mm512_store_pd( cut.dy + n, vy );
                    _mm512_store_pd( cut.ps + n, ps );
                    _mm512_store_epi64( cut.id + n, i1 );
                }
                for( ; n < nb_cuts; ++n ) {
                    TI i1 = grid.dpc_values[ cell.dpc_offset + n ];
                    Pt V = positions[ i1 ] - c0;
                    cut.dx[ n ] = V.x;
                    cut.dy[ n ] = V.y;
                    cut.id[ n ] = i1;
                    cut.ps[ n ] = dot( c0, V ) + TF( 0.5 ) * ( flags & homogeneous_weights ? norm_2_p2( V ) : norm_2_p2( V ) + w0 - weights[ i1 ] );
                }

                --nb_cuts;
                for( TI n = 0; n < nb_cuts; ++n ) {
                    if ( grid.dpc_values[ cell.dpc_offset + n ] == i0 ) {
                        cut.dx[ n ] = cut.dx[ nb_cuts ];
                        cut.dy[ n ] = cut.dy[ nb_cuts ];
                        cut.id[ n ] = cut.id[ nb_cuts ];
                        cut.ps[ n ] = cut.ps[ nb_cuts ];
                        break;
                    }
                }

                #else
                TI nb_cuts = 0;
                for( TI i1 : Span<TI>{ grid.dpc_values.data(), cell.dpc_offset, dell.dpc_offset } ) {
                    if ( i1 != i0 ) {
                        Pt V = positions[ i1 ] - c0;
                        cut.dx[ nb_cuts ] = V.x;
                        cut.dy[ nb_cuts ] = V.y;
                        cut.id[ nb_cuts ] = i1;
                        cut.ps[ nb_cuts ] = dot( c0, V ) + TF( 0.5 ) * ( flags & homogeneous_weights ? norm_2_p2( V ) : norm_2_p2( V ) + w0 - weights[ i1 ] );
                        ++nb_cuts;
                    }
                }

                #endif

                // do the cuts
                lc.plane_cut( cut.dx, cut.dy, cut.ps, cut.id, nb_cuts );

                // front
                front.init( num_cell, positions[ i0 ], weights[ i0 ] );

                // => neighbors in the same grid
                for( TI num_ng_cell : Span<TI>{ grid.ng_indices.data(), grid.ng_offsets[ num_cell + 0 ], grid.ng_offsets[ num_cell + 1 ] } )
                    front.push_without_check( num_ng_cell, grid );

                // neighbors
                while( ! front.empty() ) {
                    typename Front::Item cr = front.pop();
                    const Cell &cr_cell = grid.cells[ cr.num_cell ];

                    // if no cut is possible, we don't go further.
                    if ( ! may_cut( lc, c0, w0, cr_cell, N<flags>() ) )
                        continue;

                    // if we have diracs in cr that may cut, try them
                    const Cell &dr_cell = grid.cells[ cr.num_cell + 1 ];
                    #ifdef __AVX512F__
                    TI nb_cuts = dr_cell.dpc_offset - cr_cell.dpc_offset;
                    TI n = 0;
                    for( ; n + 8 <= nb_cuts; n += 8 ) {
                        const void *x = &positions[ 0 ].x;
                        const void *y = &positions[ 0 ].y;
                        __m512i i1 = _mm512_loadu_si512( grid.dpc_values.data() + cr_cell.dpc_offset + n );
                        __m512i m1 = _mm512_mullo_epi64( i1, _mm512_set1_epi64( 16 ) );
                        __m512d cx = _mm512_set1_pd( c0.x );
                        __m512d cy = _mm512_set1_pd( c0.y );
                        __m512d vx = _mm512_sub_pd( _mm512_i64gather_pd( m1, x, 1 ), cx );
                        __m512d vy = _mm512_sub_pd( _mm512_i64gather_pd( m1, y, 1 ), cy );
                        __m512d v2 = _mm512_add_pd( _mm512_mul_pd( vx, vx ), _mm512_mul_pd( vy, vy ) );
                        __m512d ps = _mm512_add_pd( _mm512_add_pd( _mm512_mul_pd( cx, vx ), _mm512_mul_pd( cy, vy ) ),
                                    _mm512_set1_pd( 0.5 ) * ( flags & homogeneous_weights ? v2 : _mm512_add_pd( v2, _mm512_set1_pd( w0 ) ) - _mm512_i64gather_pd( i1, weights, 8 ) ) );
                        _mm512_store_pd( cut.dx + n, vx );
                        _mm512_store_pd( cut.dy + n, vy );
                        _mm512_store_pd( cut.ps + n, ps );
                        _mm512_store_epi64( cut.id + n, i1 );
                    }
                    for( ; n < nb_cuts; ++n ) {
                        TI i1 = grid.dpc_values[ cr_cell.dpc_offset + n ];
                        Pt V = positions[ i1 ] - c0;
                        cut.dx[ n ] = V.x;
                        cut.dy[ n ] = V.y;
                        cut.id[ n ] = i1;
                        cut.ps[ n ] = dot( c0, V ) + TF( 0.5 ) * ( flags & homogeneous_weights ? norm_2_p2( V ) : norm_2_p2( V ) + w0 - weights[ i1 ] );
                    }
                    #else
                    TI nb_cuts = 0;
                    for( TI i1 : Span<TI>{ grid.dpc_values.data(), cr_cell.dpc_offset, dr_cell.dpc_offset } ) {
                        Pt V = positions[ i1 ] - c0;
                        cut.dx[ nb_cuts ] = V.x;
                        cut.dy[ nb_cuts ] = V.y;
                        cut.id[ nb_cuts ] = i1;
                        cut.ps[ nb_cuts ] = dot( c0, V ) + TF( 0.5 ) * ( flags & homogeneous_weights ? norm_2_p2( V ) : norm_2_p2( V ) + w0 - weights[ i1 ] );
                        ++nb_cuts;
                    }
                    #endif

                    lc.plane_cut( cut.dx, cut.dy, cut.ps, cut.id, nb_cuts );

                    // update the front
                    for( TI num_ng_cell : Span<TI>{ grid.ng_indices.data(), grid.ng_offsets[ cr.num_cell + 0 ], grid.ng_offsets[ cr.num_cell + 1 ] } )
                        front.push( num_ng_cell, grid );
                }

                //
                // if ( flags & ball_cut )
                //     lc.ball_cut( positions[ num_dirac ], sqrt( weights[ num_dirac ] ), num_dirac );
                // else
                //     lc.sphere_center = positions[ num_dirac ];

                //
                if ( stop_if_void_lc && lc.empty() ) {
                    err = 1;
                    break;
                }

                //
                cb( lc, i0, num_thread );
            }
        }
    } );

    return err;
}

template<class Pc>
void ZGrid<Pc>::update_the_limits( std::array<const TF *,dim> positions, const TF *weights, std::size_t nb_diracs ) {
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
void ZGrid<Pc>::fill_grid_using_zcoords( std::array<const TF *,dim> positions, const TF */*weights*/, std::size_t nb_diracs ) {
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
            if ( level == 0 )
                break;
            // ASSERT( level, "Seems not possible to have $max_diracs_per_cell considering the discretisation (some points are too close)" );
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
void ZGrid<Pc>::display_tikz( std::ostream &os, TF scale ) const {
    for( TI num_cell = 0; num_cell < grid.cells.size() - 1; ++num_cell ) {
        Pt p;
        for( int d = 0; d < dim; ++d )
            p[ d ] = grid.cells[ num_cell ].pos[ d ];

        TF a = 0, b = grid.cells[ num_cell ].size;
        switch ( dim ) {
        case 2:
            os << "\\draw ";
            os << "(" << scale * ( p[ 0 ] + a ) << "," << scale * ( p[ 1 ] + a ) << ") -- ";
            os << "(" << scale * ( p[ 0 ] + b ) << "," << scale * ( p[ 1 ] + a ) << ") -- ";
            os << "(" << scale * ( p[ 0 ] + b ) << "," << scale * ( p[ 1 ] + b ) << ") -- ";
            os << "(" << scale * ( p[ 0 ] + a ) << "," << scale * ( p[ 1 ] + b ) << ") -- ";
            os << "(" << scale * ( p[ 0 ] + a ) << "," << scale * ( p[ 1 ] + a ) << ") ;\n";
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

} // namespace sdot
