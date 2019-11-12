#include "../Geometry/Internal/ZCoords.h"
#include "../Support/StaticRange.h"
#include "../Support/RadixSort.h"
#include "../Support/ASSERT.h"
#include "../Support/Stat.h"
#include "../Support/Span.h"
#include "LGrid.h"
#include <queue>
#include <cmath>

namespace sdot {

template<class Pc>
LGrid<Pc>::LGrid( std::size_t max_diracs_per_cell ) : max_diracs_per_cell( max_diracs_per_cell ) {
    nb_final_cells = 0;
    root_cell = nullptr;
}

template<class Pc> template<int flags>
void LGrid<Pc>::update( std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs, N<flags>, bool positions_have_changed, bool weights_have_changed ) {
    if ( positions_have_changed )
        update_the_limits( positions, nb_diracs );

    if ( positions_have_changed || weights_have_changed )
        fill_the_grid( positions, weights, nb_diracs );
}

template<class Pc> template<int flags>
int LGrid<Pc>::for_each_laguerre_cell( const std::function<void( CP &, TI num, int num_thread )> &cb, const CP &starting_lc, std::array<const TF *,dim> positions, const TF *weights, TI /*nb_diracs*/, N<flags>, bool stop_if_void_lc ) {
    struct Msi { bool operator<( const Msi &that ) const { return dist > that.dist; } Pt center; const BaseCell *cell; TF dist; };
    struct CpAndNum { const SuperCell *cell; TI num; };

    //
    auto cut_lc = [&]( CP &lc, Pt c0, TF w0, TI i0, const FinalCell *dell, auto avoid_n0, TI n0 ) {
        //
        if ( dim == 3 )
            TODO;
        struct alignas(64) Cut {
            TF dx[ 128 ];
            TF dy[ 128 ];
            TF ps[ 128 ];
            CI id[ 128 ];
        };
        Cut cut;

        #ifdef __AVX512F__
        TI n1 = 0, nb_cuts = dell->nb_diracs();
        for( ; n1 + 8 <= nb_cuts; n1 += 8 ) {
            __m512d cx = _mm512_set1_pd( c0.x );
            __m512d cy = _mm512_set1_pd( c0.y );
            __m512i i1 = _mm512_loadu_si512( dell->dirac_indices + n1 );
            __m512d vx = _mm512_sub_pd( _mm512_i64gather_pd( i1, positions[ 0 ], 8 ), cx );
            __m512d vy = _mm512_sub_pd( _mm512_i64gather_pd( i1, positions[ 1 ], 8 ), cy );
            __m512d v2 = _mm512_add_pd( _mm512_mul_pd( vx, vx ), _mm512_mul_pd( vy, vy ) );
            __m512d ps = _mm512_add_pd( _mm512_add_pd( _mm512_mul_pd( cx, vx ), _mm512_mul_pd( cy, vy ) ),
                                        _mm512_set1_pd( 0.5 ) * ( flags & homogeneous_weights ? v2 : _mm512_add_pd( v2, _mm512_set1_pd( w0 ) ) - _mm512_i64gather_pd( i1, weights, 8 ) ) );
            _mm512_store_pd( cut.dx + n1, vx );
            _mm512_store_pd( cut.dy + n1, vy );
            _mm512_store_pd( cut.ps + n1, ps );
            _mm512_store_epi64( cut.id + n1, i1 );
        }
        for( ; n1 < nb_cuts; ++n1 ) {
            TI i1 = dell->dirac_indices[ n1 ];
            Pt dc = pt( positions, i1 ) - c0;
            TF w1 = weights[ i1 ];
            cut.dx[ n1 ] = dc.x;
            cut.dy[ n1 ] = dc.y;
            cut.id[ n1 ] = i1;
            cut.ps[ n1 ] = dot( c0, dc ) + TF( 0.5 ) * ( flags & homogeneous_weights ? norm_2_p2( dc ) : norm_2_p2( dc ) + w0 - w1 );
        }

        if ( avoid_n0 ) {
            --nb_cuts;
            cut.dx[ n0 ] = cut.dx[ nb_cuts ];
            cut.dy[ n0 ] = cut.dy[ nb_cuts ];
            cut.id[ n0 ] = cut.id[ nb_cuts ];
            cut.ps[ n0 ] = cut.ps[ nb_cuts ];
        }

        #else
        TI nb_cuts = 0;
        for( std::size_t n1 = 0; n1 < dell->nb_diracs(); ++n1 ) {
            if ( avoid_n0 && n1 == n0 )
                continue;
            TI i1 = dell->dirac_indices[ n1 ];
            Pt c1 = pt( positions, i1 );
            TF dw = flags & homogeneous_weights ? 0 : weights[ i1 ] - w0;
            cut.dx[ nb_cuts ] = c1.x - c0.x;
            cut.dy[ nb_cuts ] = c1.y - c0.y;
            cut.id[ nb_cuts ] = i1;
            cut.ps[ nb_cuts ] = TF( 0.5 ) * ( norm_2_p2( c1 ) - norm_2_p2( c0 ) + dw );
            ++nb_cuts;
        }
        #endif

        // do the cuts
        lc.plane_cut( { cut.dx, cut.dy }, cut.ps, cut.id, nb_cuts );
    };

    //
    int err;
    auto make_lc_from = [&]( std::priority_queue<Msi> &base_queue, std::priority_queue<Msi> &queue, CP &lc, const FinalCell *cell, const CpAndNum *path, TI path_len, int num_thread )  {
        // helper to add a cell in the queue
        auto append_msi = [&]( std::priority_queue<Msi> &queue, const BaseCell *dell, Pt cell_center ) {
            Pt dell_center = 0.5 * ( dell->bounds.min_pos + dell->bounds.max_pos );
            queue.push( Msi{ dell_center, dell, norm_2( dell_center - cell_center ) } );
        };

        // fill a first queue
        base_queue = {};
        const Pt cell_center = 0.5 * ( cell->bounds.min_pos + cell->bounds.max_pos );
        for( std::size_t num_in_path = 0; num_in_path < path_len; ++num_in_path )
            for( std::size_t i = 0; i < path[ num_in_path ].cell->nb_sub_cells(); ++i )
                if ( i != path[ num_in_path ].num )
                    append_msi( base_queue, path[ num_in_path ].cell->sub_cells[ i ], cell_center );

        // for each dirac
        for( std::size_t n0 = 0; n0 < cell->nb_diracs(); ++n0 ) {
            TI i0 = cell->dirac_indices[ n0 ];
            Pt c0 = pt( positions, i0 );
            TF w0 = flags & homogeneous_weights ? 0 : weights[ i0 ];
            lc = starting_lc;

            // cut with diracs from the same cell
            cut_lc( lc, c0, w0, i0, cell, N<1>(), n0 );

            // neighbors
            queue = base_queue;
            while ( ! queue.empty() ) {
                Msi msi = queue.top();
                queue.pop();

                // if not potential cut, we don't go further
                if ( can_be_evicted( lc, c0, w0, msi.cell->bounds, N<flags>() ) )
                    continue;

                // if final cell, do the cuts and continue the loop
                if ( msi.cell->final_cell() ) {
                    const FinalCell *dell = static_cast<const FinalCell *>( msi.cell );
                    cut_lc( lc, c0, w0, i0, dell, N<0>(), 0 );
                    continue;
                }

                // else, add sub_cells in the queue
                const SuperCell *spc = static_cast<const SuperCell *>( msi.cell );
                for( std::size_t i = 0; i < spc->nb_sub_cells(); ++i )
                    append_msi( queue, spc->sub_cells[ i ], c0 );
            }

            //
            cb( lc, i0, num_thread );
        }
    };

    if ( ! root_cell )
        return err;
    if ( root_cell->final_cell() ) {
        const FinalCell *cell =static_cast<const FinalCell *>( root_cell );
        std::priority_queue<Msi> base_queue, queue;
        CP lc;

        make_lc_from( base_queue, queue, lc, cell, nullptr, 0, 0 );
        return err;
    }

    // parallel traversal of the cells
    int nb_threads = thread_pool.nb_threads(), nb_jobs = nb_threads;
    thread_pool.execute( nb_jobs, [&]( std::size_t num_job, int num_thread ) {
        TI beg_cell = ( num_job + 0 ) * nb_final_cells / nb_jobs;
        TI end_cell = ( num_job + 1 ) * nb_final_cells / nb_jobs;
        TI end_indc = end_cell + 1;

        // path to `beg_cell`
        TI path_len = 0;
        CpAndNum path[ nb_bits_per_axis ];
        for( BaseCell *cell = root_cell; ; ++path_len ) {
            if ( cell->final_cell() )
                break;
            SuperCell *spc = static_cast<SuperCell *>( cell );
            for( std::size_t i = 0; ; ++i ) {
                BaseCell *suc = spc->sub_cells[ i ];
                if ( suc->end_ind_in_fcells > beg_cell ) {
                    path[ path_len ].cell = spc;
                    path[ path_len ].num = i;
                    cell = suc;
                    break;
                }
            }
        }

        // up to end_cell
        std::priority_queue<Msi> base_queue, queue;
        CP lc;
        while ( true ) {
            const BaseCell  *lbce = path[ path_len - 1 ].cell->sub_cells[ path[ path_len - 1 ].num ];
            const FinalCell *cell = static_cast<const FinalCell *>( lbce );
            if ( cell->end_ind_in_fcells == end_indc )
                return;

            // current cell
            make_lc_from( base_queue, queue, lc, cell, path, path_len, num_thread );

            // next one
            while ( ++path[ path_len - 1 ].num == path[ path_len - 1 ].cell->nb_sub_cells() )
                if ( --path_len == 0 )
                    return;
            while ( true ) {
                const BaseCell *tspc = path[ path_len - 1 ].cell->sub_cells[ path[ path_len - 1 ].num ];
                if ( tspc->final_cell() )
                    break;
                path[ path_len ].cell = static_cast<const SuperCell *>( tspc );
                path[ path_len ].num = 0;
                ++path_len;
            }
        }
    } );

    return err;
}

template<class Pc> template<int flags>
bool LGrid<Pc>::can_be_evicted( const CP &lc, Pt &c0, TF w0, const CellBoundsPpos<Pc> &bounds, N<flags> ) const {
    return false;
}

template<class Pc> template<int flags>
bool LGrid<Pc>::can_be_evicted( const CP &lc, Pt &c0, TF w0, const CellBoundsP0<Pc> &bounds, N<flags> ) const {
    using std::sqrt;
    using std::max;
    using std::min;
    using std::pow;
    using std::abs;

    //
    if ( flags & ball_cut ) {
        TODO;
        //        TF md = 0; // min dist pow 2
        //        for( size_t d = 0; d < dim; ++d ) {
        //            TF o = c0[ d ] - cell->min_pos[ d ];
        //            if ( o > 0 )
        //                o = max( TF( 0 ), c0[ d ] - cell->max_pos[ d ] );
        //            md += pow( o, 2 );
        //        }

        //        return md < pow( sqrt( cell->max_weight ) + sqrt( w0 ), 2 );
    }

    // for each point p in lc, can be evicted if m <= 0, with (b1=box)
    // m = max_{c1 \in b1}}( || p - c0 ||^2 - w0 - || p - c1 ||^2 + w1(q1) )
    // m = max_{c1 \in b1}}( || c0 ||^2 - || c1 ||^2 + 2 * dot( p, c1 - c0 ) - w0 + w1(q1) )
    // m = max_{c1 \in b1}}( || c0 ||^2 - w0 + w1_M - || c1 ||^2 + 2 * dot( p, c1 - c0 ) )
    TF cc = norm_2_p2( c0 ) - w0 + bounds.max_weight;
    for( TI num_lc_point = 0; num_lc_point < lc.nb_nodes(); ++num_lc_point ) {
        Pt p = lc.node( num_lc_point ).pos();
        Pt c1 = max( min( p, bounds.max_pos ), bounds.min_pos );
        if ( cc + 2 * dot( p, c1 - c0 ) > norm_2_p2( c1 ) )
            return false;
    }

    // Order 1, 2D:
    // m = max_{c1 \in b1}}( || c0 ||^2 + w1_0 - w0 - c1_x^2 - c1_y^2 + 2 * p_x * ( c1_x - c0_x ) + 2 * p_y * ( c1_y - c0_y ) )
    // => c1 = clamp(  )

    //    #ifdef __AVX512F__
    //    if ( lc.nb_nodes() <= 8 ) {
    //        __m512d p_x = _mm512_load_pd( &lc.node( 0 ).x );
    //        __m512d p_y = _mm512_load_pd( &lc.node( 0 ).y );
    //        __m512d c0x = _mm512_set1_pd( c0.x );
    //        __m512d c0y = _mm512_set1_pd( c0.y );
    //        __m512d d0x = _mm512_sub_pd( p_x, c0x );
    //        __m512d d0y = _mm512_sub_pd( p_y, c0y );

    //        const TF s = TF( 0.5 ) * cr_cell.size;
    //        const Pt C = cr_cell.pos + s;

    //        // | p - c1 | => max( 0, abs( p - C ) - s )
    //        __m512d sp = _mm512_set1_pd( s );
    //        __m512d zp = _mm512_setzero_pd();
    //        __m512d d1x = _mm512_max_pd( zp, _mm512_sub_pd( _mm512_abs_pd( _mm512_sub_pd( p_x, _mm512_set1_pd( C.x ) ) ), sp ) );
    //        __m512d d1y = _mm512_max_pd( zp, _mm512_sub_pd( _mm512_abs_pd( _mm512_sub_pd( p_y, _mm512_set1_pd( C.y ) ) ), sp ) );

    //        // | p - c0 |^2, | p - c1 |^2
    //        __m512d d02 = _mm512_add_pd( _mm512_mul_pd( d0x, d0x ), _mm512_mul_pd( d0y, d0y ) );
    //        __m512d d12 = _mm512_add_pd( _mm512_mul_pd( d1x, d1x ), _mm512_mul_pd( d1y, d1y ) );

    //        if ( flags & homogeneous_weights ) {
    //            // | p - c1 |^2 < | p - c0 |^2
    //            int n = _mm512_cmp_pd_mask( d12, d02, _CMP_LT_OQ );
    //            if ( n & ( ( 1 << lc.nb_nodes() ) - 1 ) )
    //                return true;
    //        } else {
    //            // | p - c1 |^2 - | p - c0 |^2 < w1^2 - w0^2
    //            int n = _mm512_cmp_pd_mask( _mm512_sub_pd( d12, d02 ), _mm512_set1_pd( pow( max_weight, 2 ) - pow( w0, 2 ) ), _CMP_LT_OQ );
    //            if ( n & ( ( 1 << lc.nb_nodes() ) - 1 ) )
    //                return true;
    //        }
    //    } else {
    //        const Pt C = cr_cell.pos + TF( 0.5 ) * cr_cell.size;
    //        const TF s = sqrt( TF( 0.5 ) ) * cr_cell.size;
    //        for( TI num_lc_point = 0; num_lc_point < lc.nb_nodes(); ++num_lc_point ) {
    //            Pt p = lc.node( num_lc_point ).pos();
    //            TF r2 = norm_2_p2( p - c0 );
    //            if ( ( flags & homogeneous_weights ) == 0 )
    //                r2 += max_weight - w0;
    //            if ( pow( norm_2( p - C ) - s, 2 ) < r2 )
    //                return true;
    //        }
    //    }
    //    #else
    //    const TF s = TF( 0.5 ) * norm_2( cell->max_pos - cell->min_pos );
    //    const Pt C = TF( 0.5 ) * ( cell->min_pos + cell->max_pos );
    //    for( TI num_lc_point = 0; num_lc_point < lc.nb_nodes(); ++num_lc_point ) {
    //        Pt p = lc.node( num_lc_point ).pos();
    //        TF r2 = norm_2_p2( p - c0 );
    //        //        if ( ( flags & homogeneous_weights ) == 0 )
    //        //            r2 += max_weight - w0;
    //        if ( pow( norm_2( p - C ) - s, 2 ) < r2 )
    //            return true;
    //    }
    //    #endif

    return true;
}

template<class Pc>
void LGrid<Pc>::write_to_stream( std::ostream &os ) const {
    write_to_stream( os, root_cell, {} );
}


template<class Pc>
void LGrid<Pc>::write_to_stream( std::ostream &os, BaseCell *cell, std::string sp ) const {
    if ( ! cell ) {
        os << sp << "null";
        return;
    }

    cell->min_pos.write_to_stream( os << sp << "mip=" );
    cell->max_pos.write_to_stream( os << " map=" );
    os << " e=" << cell->end_ind_in_fcells;
    if ( cell->super_cell() ) {
        const SuperCell *sc = static_cast<const SuperCell *>( cell );
        os << " nb_sub=" << sc->nb_sub_cells();
        for( std::size_t i = 0; i < sc->nb_sub_cells(); ++i )
            write_to_stream( os << "\n", sc->sub_cells[ i ], sp + "  " );
    }
    if ( cell->final_cell() ) {
        const FinalCell *sc = static_cast<const FinalCell *>( cell );
        os << " nb_diracs=" << sc->nb_diracs();
    }
}

template<class Pc>
void LGrid<Pc>::update_the_limits( std::array<const TF *,dim> positions, TI nb_diracs ) {
    using std::min;
    using std::max;

    // min/max
    for( std::size_t d = 0; d < dim; ++d ) {
        min_point[ d ] = + std::numeric_limits<TF>::max();
        max_point[ d ] = - std::numeric_limits<TF>::max();
    }

    for( std::size_t num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
        for( std::size_t d = 0; d < dim; ++d ) {
            min_point[ d ] = min( min_point[ d ], positions[ d ][ num_dirac ] );
            max_point[ d ] = max( max_point[ d ], positions[ d ][ num_dirac ] );
        }
    }

    // grid size
    grid_length = 0;
    for( std::size_t d = 0; d < dim; ++d )
        grid_length = max( grid_length, max_point[ d ] - min_point[ d ] );
    grid_length *= 1 + std::numeric_limits<TF>::epsilon();

    step_length = grid_length / ( TZ( 1 ) << nb_bits_per_axis );
    inv_step_length = TF( 1 ) / step_length;
}


template<class Pc>
void LGrid<Pc>::fill_the_grid( std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs ) {
    static_assert( sizeof( TZ ) >= sizeof_zcoords, "zcoords types (TZ) is not large enough" );
    using LocalSolver = typename CellBounds::LocalSolver;

    using std::round;
    using std::ceil;
    using std::pow;
    using std::min;
    using std::max;

    // get zcoords for each dirac
    znodes_keys.reserve( 2 * nb_diracs );
    znodes_inds.reserve( 2 * nb_diracs );
    make_znodes<nb_bits_per_axis>( znodes_keys.data(), znodes_inds.data(), positions, nb_diracs, min_point, inv_step_length );

    // sorting w.r.t. zcoords
    std::pair<TZ *,TI *> sorted_znodes = radix_sort(
        std::make_pair( znodes_keys.data() + nb_diracs, znodes_inds.data() + nb_diracs ),
        std::make_pair( znodes_keys.data(), znodes_inds.data() ),
        nb_diracs,
        N<dim*nb_bits_per_axis>(),
        rs_tmps
    );

    // helpers to update level info
    struct LevelInfo {
        void        clr                    () { num_sub_cell = 0; nb_sub_cells = 0; ls.clr(); }

        TI          num_sub_cell;          ///<
        TI          nb_sub_cells;          ///<
        BaseCell   *sub_cells[ 1 << dim ]; ///<
        LocalSolver ls;
    };

    LevelInfo level_info[ nb_bits_per_axis + 1 ];
    for( LevelInfo &l : level_info )
        l.clr();

    // get the cells zcoords and indices (offsets in dpc_indices) + dpc_indices
    int level = 0;
    TZ prev_z = 0;
    root_cell = nullptr;
    nb_final_cells = 0;
    for( TI index = max_diracs_per_cell; ; ) {
        auto push_cell = [&]( TI l ) {
            TZ old_prev_z = prev_z;
            prev_z += TZ( 1 ) << dim * level;

            // beg/end of cells to push (indices in sorted_znodes)
            TI beg_ind_zn = l, len_ind_nz = 0;
            for( TI n = index - max_diracs_per_cell; n < l; ++n ) {
                if ( sorted_znodes.first[ n ] >= old_prev_z ) {
                    beg_ind_zn = n;
                    for( ; ; ++n  ) {
                        if ( n == l || sorted_znodes.first[ n ] >= prev_z ) {
                            len_ind_nz = n - beg_ind_zn;
                            break;
                        }
                    }
                    break;
                }
            }

            //
            index += len_ind_nz;

            // prepare a new cell, register it in corresponding level_info
            LevelInfo *li = level_info + level;
            BaseCell *cell = nullptr;
            if ( len_ind_nz ) {
                cell = reinterpret_cast<BaseCell *>( mem_pool.allocate( sizeof( BaseCell ) + len_ind_nz * sizeof( TI ) ) );
                FinalCell *fcell = static_cast<FinalCell *>( cell );
                cell->end_ind_in_fcells = ++nb_final_cells;
                cell->nb_sub_items = len_ind_nz;

                // store diracs indices, get bounds
                LocalSolver ls;
                ls.clr();
                for( TI i = 0; i < len_ind_nz; ++i ) {
                    TI ind = sorted_znodes.second[ beg_ind_zn + i ];
                    fcell->dirac_indices[ i ] = ind;

                    ls.push( pt( positions, ind ), weights[ ind ] );
                }

                ls.store_to( cell->bounds );

                //
                li->sub_cells[ li->nb_sub_cells++ ] = cell;
                li->ls.push( ls );
            }

            // multilevel
            for( std::size_t sl = level; ; ++sl ) {
                // coarser level ?
                if ( sl == nb_bits_per_axis ) {
                    root_cell = cell;
                    break;
                }

                // if the sub cells are not finished, stay in this level
                if ( li->num_sub_cell < ( 1 << dim ) - 1 ) {
                    ++li->num_sub_cell;
                    break;
                }

                // else, make a new super cell
                cell = nullptr;
                LevelInfo *oli = li++;
                if ( oli->nb_sub_cells ) {
                    if ( oli->nb_sub_cells > 1 ) {
                        cell = reinterpret_cast<BaseCell *>( mem_pool.allocate( sizeof( BaseCell ) + oli->nb_sub_cells * sizeof( BaseCell * ) ) );
                        cell->end_ind_in_fcells = nb_final_cells;
                        cell->nb_sub_items = - oli->nb_sub_cells;

                        SuperCell *scell = static_cast<SuperCell *>( cell );
                        for( std::size_t i = 0; i < oli->nb_sub_cells; ++i )
                            scell->sub_cells[ i ] = oli->sub_cells[ i ];

                        //
                        oli->ls.store_to( cell->bounds );
                    } else {
                        cell = oli->sub_cells[ 0 ];
                    }

                    //
                    li->sub_cells[ li->nb_sub_cells++ ] = cell;
                    li->ls.push( oli->ls );
                }

                // and reset the previous level
                oli->clr();
            }
        };

        // last cell(s)
        if ( index >= nb_diracs ) {
            while ( prev_z < ( TZ( 1 ) << dim * nb_bits_per_axis ) ) {
                for( ; ; ++level ) {
                    TZ m = TZ( 1 ) << dim * ( level + 1 );
                    if ( level == nb_bits_per_axis || prev_z & ( m - 1 ) ) {
                        push_cell( nb_diracs );
                        break;
                    }
                }
            }
            break;
        }

        // level too high ?
        for( ; ; --level ) {
            TZ m = TZ( 1 ) << dim * ( level + 1 );
            if ( sorted_znodes.first[ index ] >= prev_z + m )
                break;
            if ( level == 0 )
                break;
        }

        // look for a level before the one that will take the $max_diracs_per_cell next points or that will lead to an illegal cell
        for( ; ; ++level ) {
            TZ m = TZ( 1 ) << dim * ( level + 1 );
            if ( sorted_znodes.first[ index ] < prev_z + m || ( prev_z & ( m - 1 ) ) ) {
                push_cell( index );
                break;
            }
        }
    }

    //
    if ( CellBounds::need_phase_1 && root_cell ) {
        BaseCell *path[ nb_bits_per_axis ];
        update_cell_bounds_phase_1( positions, weights, root_cell, path, 0 );
    }
}

template<class Pc>
void LGrid<Pc>::update_cell_bounds_phase_1( std::array<const TF *,dim> positions, const TF *weights, BaseCell *cell, BaseCell **path, int level ) {
    path[ level ] = cell;

    if ( cell->final_cell() ) {
        FinalCell *fc = static_cast<FinalCell *>( cell );
        for( size_t n = 0; n < fc->nb_diracs(); ++n ) {
            TI ind = fc->dirac_indices[ n ];
            Pt pos = pt( positions, ind );
            TF w = weights[ ind ];

            for( size_t l = 0; l <= level; ++l )
                path[ l ]->bounds.push( pos, w );
        }
        return;
    }

    if ( cell->super_cell() ) {
        SuperCell *sc = static_cast<SuperCell *>( cell );
        for( size_t i = 0; i < sc->nb_sub_cells(); ++i )
            update_cell_bounds_phase_1( positions, weights, sc->sub_cells[ i ], path, level + 1 );
        return;
    }

    TODO;
}

template<class Pc>
void LGrid<Pc>::display_tikz( std::ostream &os, TF scale ) const {
    //    for( TI num_cell = 0; num_cell < cells.size() - 1; ++num_cell ) {
    //        Pt p;
    //        for( int d = 0; d < dim; ++d )
    //            p[ d ] = cells[ num_cell ].pos[ d ];

    //        TF a = 0, b = cells[ num_cell ].size;
    //        switch ( dim ) {
    //        case 2:
    //            os << "\\draw ";
    //            os << "(" << scale * ( p[ 0 ] + a ) << "," << scale * ( p[ 1 ] + a ) << ") -- ";
    //            os << "(" << scale * ( p[ 0 ] + b ) << "," << scale * ( p[ 1 ] + a ) << ") -- ";
    //            os << "(" << scale * ( p[ 0 ] + b ) << "," << scale * ( p[ 1 ] + b ) << ") -- ";
    //            os << "(" << scale * ( p[ 0 ] + a ) << "," << scale * ( p[ 1 ] + b ) << ") -- ";
    //            os << "(" << scale * ( p[ 0 ] + a ) << "," << scale * ( p[ 1 ] + a ) << ") ;\n";
    //            break;
    //        case 3:
    //            TODO;
    //            break;
    //        default:
    //            TODO;
    //        }
    //    }
    TODO;
}

template<class Pc>
void LGrid<Pc>::display( VtkOutput &vtk_output, std::array<const TF *,dim> positions, const TF *weights, int disp_weights ) const {
    if ( root_cell )
        display( vtk_output, positions, weights, root_cell, disp_weights );
}

template<class Pc>
void LGrid<Pc>::display( VtkOutput &vtk_output, std::array<const TF *,dim> positions, const TF *weights, BaseCell *cell, int disp_weights ) const {
    Pt a = cell->bounds.min_pos, b = cell->bounds.max_pos;
    std::vector<Point3<TF>> pts = {
        Point3<TF>{ a[ 0 ], a[ 1 ], 0 },
        Point3<TF>{ b[ 0 ], a[ 1 ], 0 },
        Point3<TF>{ b[ 0 ], b[ 1 ], 0 },
        Point3<TF>{ a[ 0 ], b[ 1 ], 0 },
    };
    if ( disp_weights ) {
        for( Point3<TF> &pt : pts )
            pt[ 2 ] = cell->bounds.get_w( { pt[ 0 ], pt[ 1 ] } );
    }
    vtk_output.add_polygon( pts );

    if ( cell->super_cell() ) {
        const SuperCell *sc = static_cast<const SuperCell *>( cell );
        for( std::size_t i = 0; i < sc->nb_sub_cells(); ++i )
            display( vtk_output, positions, weights, sc->sub_cells[ i ], disp_weights );
    }

    if ( cell->final_cell() ) {
        const FinalCell *sc = static_cast<const FinalCell *>( cell );
        if ( disp_weights )
            for( std::size_t i = 0; i < sc->nb_diracs(); ++i )
                vtk_output.add_point( Point3<TF>{ positions[ 0 ][ sc->dirac_indices[ i ] ], positions[ 1 ][ sc->dirac_indices[ i ] ], weights[ sc->dirac_indices[ i ] ] } );
        else
            for( std::size_t i = 0; i < sc->nb_diracs(); ++i )
                vtk_output.add_point( pt( positions, sc->dirac_indices[ i ] ) );
    }
}

} // namespace sdot
