#include "../Geometry/Internal/ZCoords.h"
#include "../Support/StaticRange.h"
#include "../Support/RadixSort.h"
#include "../Support/ASSERT.h"
#include "../Support/Stat.h"
#include "../Support/Span.h"
#include "LGrid.h"
#include <cmath>

namespace sdot {

template<class Pc>
LGrid<Pc>::LGrid( std::size_t max_diracs_per_cell ) : max_diracs_per_cell( max_diracs_per_cell ) {
    max_diracs_per_sst = std::numeric_limits<TI>::max();
    nb_final_cells     = 0;
    nb_cb_calls        = 0;
    root_cell          = nullptr;
}

template<class Pc>
void LGrid<Pc>::construct( const Dirac *diracs, TI nb_diracs ) {
    construct( [&]( const Cb &cb ) {
        cb( diracs, nb_diracs, true );
    } );
}

template<class Pc>
void LGrid<Pc>::construct( const std::function<void(const Cb &cb)> &f ) {
    // get min/max of positions + positions and weight pointers (if possible to use them)
    get_grid_dims_and_dirac_ptrs( f );

    // get limits of sub-structures (parts that can be saved/loaded)
    compute_sst_limits( f );

    // create the cells + first phase of bounds update
    make_the_cells( f );

    // second phase of bounds update (if necessary)
    if ( CellBounds::need_phase_1 && root_cell ) {
        BaseCell *path[ nb_bits_per_axis ];
        update_cell_bounds_phase_1( root_cell, path, 0 );
    }
}

template<class Pc>
void LGrid<Pc>::update_grid_wrt_weights() {
    if ( root_cell ) {
        // tmp storage for multi-level information
        LocalSolver local_solvers[ nb_bits_per_axis ];
        update_grid_wrt_weights_rec( root_cell, local_solvers, 0 );
    }

    // second phase of bounds update (if necessary)
    if ( CellBounds::need_phase_1 && root_cell ) {
        BaseCell *path[ nb_bits_per_axis ];
        update_cell_bounds_phase_1( root_cell, path, 0 );
    }
}

template<class Pc>
void LGrid<Pc>::update_grid_wrt_weights_rec( BaseCell *cell, LocalSolver *local_solvers, int level ) {
    local_solvers[ level ].clr();

    if ( SuperCell *sc = cell->super_cell() ) {
        for( std::size_t i = 0; i < sc->nb_sub_cells(); ++i ) {
            update_grid_wrt_weights_rec( sc->sub_cells[ i ], local_solvers, level + 1 );
            local_solvers[ level ].push( local_solvers[ level + 1 ] );
        }
    } else if ( FinalCell *fc = cell->final_cell() ) {
        for( std::size_t i = 0; i < fc->nb_diracs(); ++i )
            local_solvers[ level ].push( fc->diracs[ i ].pos, fc->diracs[ i ].weight );
    } else {
        TODO;
    }

    local_solvers[ level ].store_to( cell->bounds );
}

template<class Pc>
void LGrid<Pc>::get_grid_dims_and_dirac_ptrs( const std::function<void(const Cb &cb)> &f ) {
    using std::min;
    using std::max;

    // reset
    nb_diracs_tot = 0;
    nb_cb_calls = 0;

    // traversal
    min_point = + std::numeric_limits<TF>::max();
    max_point = - std::numeric_limits<TF>::max();
    use_dirac_pns = true;
    dirac_pns.clear();
    f( [&]( const Dirac *diracs, TI nb_diracs, bool ptrs_survive_the_call ) {
        for( std::size_t num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
            const Dirac &dirac = diracs[ num_dirac ];
            min_point = min( min_point, dirac.pos );
            max_point = max( max_point, dirac.pos );
        }

        if ( ptrs_survive_the_call )
            dirac_pns.push_back( { diracs, nb_diracs } );
        else
            use_dirac_pns = false;

        nb_diracs_tot += nb_diracs;
        ++nb_cb_calls;
    } );

    // grid size
    grid_length = 0;
    for( std::size_t d = 0; d < dim; ++d )
        grid_length = max( grid_length, max_point[ d ] - min_point[ d ] );
    grid_length *= 1 + std::numeric_limits<TF>::epsilon();

    step_length = grid_length / ( TZ( 1 ) << nb_bits_per_axis );
    inv_step_length = TF( 1 ) / step_length;
}

template<class Pc>
void LGrid<Pc>::compute_sst_limits( const std::function<void(const Cb &cb)> &/*f*/ ) {
    if ( nb_diracs_tot <= max_diracs_per_sst ) {
        sst_limits = { SstLimits{ TZ( 0 ), TZ( 1 ) << dim * nb_bits_per_axis, nb_diracs_tot } };
        return;
    }

    // => get the subdivisions
    P( nb_diracs_tot, max_diracs_per_sst );
    TODO;
}


template<class Pc>
void LGrid<Pc>::make_the_cells( const std::function<void(const Cb &cb)> &/*f*/ ) {
    static_assert( sizeof( TZ ) >= sizeof_zcoords, "TZ (zcoords type) is not large enough" );

    // for each sub-structure
    mem_pool.clear();
    nb_final_cells = 0;
    root_cell = nullptr;
    for( const SstLimits &sst : sst_limits ) {
        // get zcoords for diracs inside the limits (znodes_keys and znodes_inds in this case)
        if ( use_dirac_pns ) {
            if ( dirac_pns.size() == 1 ) {
                // update znodes_xxx buffers
                if ( sst_limits.size() > 1 )
                    make_znodes_with_1ppwn_ssst( sst, dirac_pns[ 0 ].diracs, dirac_pns[ 0 ].nb_diracs );
                else
                    make_znodes_with_1ppwn_1sst( dirac_pns[ 0 ].diracs, dirac_pns[ 0 ].nb_diracs );

                // use znodes_xxx buffers to make the cells for this sst (maybe with unfinished bounds at this stage)
                make_the_cells_for( sst, dirac_pns[ 0 ] );
            } else {
                TODO;
            }
        } else {
            TODO;
        }

    }
}

template<class Pc>
void LGrid<Pc>::make_znodes_with_1ppwn_1sst( const Dirac *diracs, TI nb_diracs ) {
    znodes_keys.reserve( 2 * nb_diracs );
    znodes_inds.reserve( 2 * nb_diracs );

    // 1 ppwm => 1 index is enough to find the corresponding dirac
    // 1 sst => no need to test if the dirac is inside => num in znode_xxx is = to num to positions
    TI nb_threads = thread_pool.nb_threads(), nb_jobs = nb_threads, offset = 0;
    thread_pool.execute( nb_jobs, [&]( TI num_job, int /*num_thread*/ ) {
        TI beg = ( num_job + 0 ) * nb_diracs / nb_jobs;
        TI end = ( num_job + 1 ) * nb_diracs / nb_jobs;
        for( TI num_dirac = beg; num_dirac < end; ++num_dirac ) {
            TZ zcoords = zcoords_for<TZ,nb_bits_per_axis>( diracs[ num_dirac ].pos, min_point, inv_step_length );
            znodes_keys[ num_dirac ] = zcoords;
            znodes_inds[ num_dirac ] = num_dirac;
        }
    } );
}

template<class Pc>
void LGrid<Pc>::make_znodes_with_1ppwn_ssst( const SstLimits &/*sst*/, const Dirac */*diracs*/, TI /*nb_diracs*/ ) {
    //    znodes_keys.reserve( 2 * sst.nb_diracs );
    //    znodes_inds.reserve( 2 * sst.nb_diracs );

    //    TI nb_threads = thread_pool.nb_threads(), nb_jobs = nb_threads, off = 0;
    //    std::vector<TI> offsets( nb_jobs, 0 );
    //    TI offset = 0;
    //    // TODO: optimization if only 1 sst
    //    f( [&]( const Pt *positions, const TF */*weights*/, TI nb_diracs, bool ptrs_survive_the_call ) {
    //        // nb diracs per thread
    //        thread_pool.execute( nb_jobs, [&]( TI num_job, int num_thread ) {
    //            TI beg = ( num_job + 0 ) * nb_diracs / nb_jobs;
    //            TI end = ( num_job + 1 ) * nb_diracs / nb_jobs;
    //            for( TI index = beg; index < end; ++index ) {
    //                TZ zcoords = zcoords_for<TZ,nb_bits_per_axis>( positions[ index ], min_point, inv_step_length );
    //                if ( zcoords >= sst.beg_zcoords && zcoords < sst.end_zcoords )
    //                    ++offsets[ num_thread ];
    //            }
    //        } );

    //        // offset per thread
    //        for( TI i = 0; i < nb_threads; ++i ) {
    //            TI size = offsets[ i ];
    //            offsets[ i ] = offset;
    //            offset += size;
    //        }

    //        //
    //        thread_pool.execute( nb_jobs, [&]( TI num_job, int num_thread ) {
    //            TI beg = ( num_job + 0 ) * nb_diracs / nb_jobs;
    //            TI end = ( num_job + 1 ) * nb_diracs / nb_jobs;
    //            for( TI index = beg; index < end; ++index ) {
    //                TZ zcoords = zcoords_for<TZ,nb_bits_per_axis>( positions[ index ], min_point, inv_step_length );
    //                if ( zcoords >= sst.beg_zcoords && zcoords < sst.end_zcoords ) {
    //                    TI off = offsets[ num_thread ]++;
    //                    zcoords[ off ] = zcoords;
    //                    indices[ off ] = index;
    //                }
    //            }
    //        } );
    //    } );
    TODO;
}


template<class Pc> template<class Ps>
void LGrid<Pc>::make_the_cells_for( const SstLimits &sst, Ps ps ) {
    struct          TmpLevelInfo                {
        void        clr                         () { num_sub_cell = 0; nb_sub_cells = 0; ls.clr(); }
        BaseCell   *sub_cells[ 1 << dim ];      ///<
        TI          num_sub_cell;               ///<
        TI          nb_sub_cells;               ///<
        LocalSolver ls;
    };

    // tmp storage for multi-level information
    TmpLevelInfo level_info[ nb_bits_per_axis + 1 ];
    for( TmpLevelInfo &l : level_info )
        l.clr();

    auto *pinds = znodes_seconds( ps );

    // sorting w.r.t. zcoords
    auto sorted_znodes = radix_sort(
        std::make_pair( znodes_keys.data() + sst.nb_diracs, pinds + sst.nb_diracs ),
        std::make_pair( znodes_keys.data(), pinds ),
        sst.nb_diracs,
        N<dim*nb_bits_per_axis>(),
        rs_tmps
    );

    // get the cells zcoords and indices (offsets in dpc_indices) + dpc_indices
    int level = sst.level;
    TZ prev_z = sst.beg_zcoords;
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
            TmpLevelInfo *li = level_info + level;
            BaseCell *cell = nullptr;
            if ( len_ind_nz ) {
                FinalCell *fcell = FinalCell::allocate( mem_pool, len_ind_nz );
                fcell->end_ind_in_fcells = ++nb_final_cells;

                // store diracs indices, get bounds
                LocalSolver ls;
                ls.clr();
                for( TI i = 0; i < len_ind_nz; ++i ) {
                    const Dirac &dirac = get_dirac( ps, sorted_znodes.second[ beg_ind_zn + i ] );
                    fcell->diracs[ i ] = dirac;

                    ls.push( dirac.pos, dirac.weight );
                }

                ls.store_to( fcell->bounds );

                //
                li->sub_cells[ li->nb_sub_cells++ ] = fcell;
                li->ls.push( ls );

                cell = fcell;
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
                TmpLevelInfo *oli = li++;
                if ( oli->nb_sub_cells ) {
                    if ( oli->nb_sub_cells > 1 ) {
                        SuperCell *scell = SuperCell::allocate( mem_pool, oli->nb_sub_cells );
                        scell->end_ind_in_fcells = nb_final_cells;

                        for( std::size_t i = 0; i < oli->nb_sub_cells; ++i )
                            scell->sub_cells[ i ] = oli->sub_cells[ i ];

                        //
                        oli->ls.store_to( scell->bounds );

                        cell = scell;
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
        if ( index >= sst.nb_diracs ) {
            while ( prev_z < ( TZ( 1 ) << dim * nb_bits_per_axis ) ) {
                for( ; ; ++level ) {
                    TZ m = TZ( 1 ) << dim * ( level + 1 );
                    if ( level == nb_bits_per_axis || prev_z & ( m - 1 ) ) {
                        push_cell( sst.nb_diracs );
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
}

template<class Pc> template<int avoid_n0,int flags>
void LGrid<Pc>::cut_lc( CP &lc, Pt c0, TF w0, FinalCell *dell, N<avoid_n0>, TI n0, N<flags> ) const {
    //
    if ( dim == 3 )
        TODO;
    struct alignas(64) Cut {
        TF     dx[ 128 ];
        TF     dy[ 128 ];
        TF     ps[ 128 ];
        Dirac *id[ 128 ];
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
        Dirac &d1 = dell->diracs[ n1 ];
        Pt c1 = d1.pos;
        TF dw = flags & homogeneous_weights ? 0 : d1.weight - w0;
        cut.dx[ nb_cuts ] = c1.x - c0.x;
        cut.dy[ nb_cuts ] = c1.y - c0.y;
        cut.id[ nb_cuts ] = &d1;
        cut.ps[ nb_cuts ] = TF( 0.5 ) * ( norm_2_p2( c1 ) - norm_2_p2( c0 ) - dw );
        ++nb_cuts;
    }
    #endif

    // do the cuts
    lc.plane_cut( { cut.dx, cut.dy }, cut.ps, cut.id, nb_cuts );
}


template<class Pc> template<int flags>
void LGrid<Pc>::make_lcs_from( const std::function<void( CP &, Dirac &dirac, int num_thread )> &cb, std::priority_queue<LGrid::Msi> &base_queue, std::priority_queue<LGrid::Msi> &queue, LGrid::CP &lc, FinalCell *cell, const LGrid::CpAndNum *path, LGrid::TI path_len, int num_thread, N<flags>, const CP &starting_lc ) const {
    // helper to add a cell in the queue
    auto append_msi = [&]( std::priority_queue<Msi> &queue, BaseCell *dell, Pt cell_center ) {
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
        Dirac &d0 = cell->diracs[ n0 ];
        TF w0 = flags & homogeneous_weights ? 0 : d0.weight;
        Pt c0 = d0.pos;
        lc = starting_lc;

        // cut with diracs from the same cell
        cut_lc( lc, c0, w0, cell, N<1>(), n0, N<flags>() );

        // neighbors
        queue = base_queue;
        while ( ! queue.empty() ) {
            Msi msi = queue.top();
            queue.pop();

            // if not potential cut, we don't go further
            if ( can_be_evicted( lc, c0, w0, msi.cell->bounds, N<flags>() ) )
                continue;

            // if final cell, do the cuts and continue the loop
            if ( FinalCell *dell = msi.cell->final_cell() ) {
                cut_lc( lc, c0, w0, dell, N<0>(), 0, N<flags>() );
                continue;
            }

            // else, add sub_cells in the queue
            const SuperCell *spc = static_cast<const SuperCell *>( msi.cell );
            for( std::size_t i = 0; i < spc->nb_sub_cells(); ++i )
                append_msi( queue, spc->sub_cells[ i ], c0 );
        }

        //
        cb( lc, d0, num_thread );
    }
}

template<class Pc>
int LGrid<Pc>::for_each_laguerre_cell( const std::function<void( CP &, Dirac &dirac, int num_thread )> &cb, const CP &starting_lc, TraversalFlags traversal_flags ) {
    constexpr int flags = 0;
    int err;

    // parallel traversal of the cells
    int nb_threads = thread_pool.nb_threads(), nb_jobs = nb_threads;
    thread_pool.execute( nb_jobs, [&]( std::size_t num_job, int num_thread ) {
        TI beg_cell = ( num_job + 0 ) * nb_final_cells / nb_jobs;
        TI end_cell = ( num_job + 1 ) * nb_final_cells / nb_jobs;
        std::priority_queue<Msi> base_queue, queue;
        CP lc;

        for_each_final_cell_mono_thr( [&]( FinalCell &cell, CpAndNum *path, TI path_len ) {
            make_lcs_from( cb, base_queue, queue, lc, &cell, path, path_len, num_thread, N<flags>(), starting_lc );
        }, beg_cell, end_cell );
    } );

    //
    if ( traversal_flags.mod_weights )
        update_grid_wrt_weights();

    return err;
}


template<class Pc>
void LGrid<Pc>::for_each_final_cell_mono_thr( const std::function<void( FinalCell &cell, CpAndNum *path, TI path_len )> &f, TI beg_cell, TI end_cell ) {
    if ( ! root_cell )
        return;

    if ( FinalCell *cell = root_cell->final_cell() ) {
        if ( beg_cell < end_cell ) {
            f( *cell, nullptr, 0 );
        }
        return;
    }

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
    TI end_indc = end_cell + 1;
    while ( true ) {
        BaseCell  *lbce = path[ path_len - 1 ].cell->sub_cells[ path[ path_len - 1 ].num ];
        FinalCell *cell = static_cast<FinalCell *>( lbce );
        if ( cell->end_ind_in_fcells == end_indc )
            return;

        // call
        f( *cell, path, path_len );

        // next one
        while ( ++path[ path_len - 1 ].num == path[ path_len - 1 ].cell->nb_sub_cells() )
            if ( --path_len == 0 )
                return;
        while ( true ) {
            BaseCell *tspc = path[ path_len - 1 ].cell->sub_cells[ path[ path_len - 1 ].num ];
            if ( tspc->final_cell() )
                break;
            path[ path_len ].cell = static_cast<SuperCell *>( tspc );
            path[ path_len ].num = 0;
            ++path_len;
        }
    }
}


template<class Pc>
void LGrid<Pc>::for_each_final_cell( const std::function<void( FinalCell &cell, int num_thread )> &f, TraversalFlags traversal_flags ) {
    // parallel traversal of the cells
    int nb_threads = thread_pool.nb_threads(), nb_jobs = nb_threads;
    thread_pool.execute( nb_jobs, [&]( std::size_t num_job, int num_thread ) {
        TI beg_cell = ( num_job + 0 ) * nb_final_cells / nb_jobs;
        TI end_cell = ( num_job + 1 ) * nb_final_cells / nb_jobs;
        for_each_final_cell_mono_thr( [&]( FinalCell &cell, CpAndNum */*path*/, TI /*path_len*/ ) {
            f( cell, num_thread );
        }, beg_cell, end_cell );
    } );

    //
    if ( traversal_flags.mod_weights )
        update_grid_wrt_weights();
}

template<class Pc>
void LGrid<Pc>::for_each_dirac( const std::function<void( Dirac &, int )> &f, TraversalFlags traversal_flags ) {
    for_each_final_cell( [&]( FinalCell &cell, int num_thread ) {
        for( std::size_t i = 0; i < cell.nb_diracs(); ++i )
            f( cell.diracs[ i ], num_thread );
    }, traversal_flags );
}

template<class Pc> template<int flags>
bool LGrid<Pc>::can_be_evicted( const CP &lc, Pt &c0, TF w0, const CellBoundsPpos<Pc> &bounds, N<flags> ) const {
    if ( flags & ball_cut ) {
        TODO;
    }

    // m = max_{c1 \in b1}}( || c0 ||^2 - w0 + w1_M - || c1 ||^2 + 2 * dot( p, c1 - c0 ) + dot( c1, w1_D ) )
    // der => 2 * c1_x = 2 * p_x + w1_D_x
    TF cc = norm_2_p2( c0 ) - w0 + bounds.poly_weight[ 0 ];
    for( TI num_lc_point = 0; num_lc_point < lc.nb_nodes(); ++num_lc_point ) {
        Pt p = lc.node( num_lc_point ).pos(), o = p;
        for( std::size_t d = 0; d < dim; ++d )
            o[ d ] += 0.5 * bounds.poly_weight[ d + 1 ];
        Pt c1 = max( min( o, bounds.max_pos ), bounds.min_pos );

        TF dd = 0;
        for( std::size_t d = 0; d < dim; ++d )
            dd += c1[ d ] * bounds.poly_weight[ d + 1 ];
        if ( cc + 2 * dot( p, c1 - c0 ) + dd > norm_2_p2( c1 ) )
            return false;
    }

    return true;
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
void LGrid<Pc>::update_cell_bounds_phase_1( BaseCell *cell, BaseCell **path, int level ) {
    path[ level ] = cell;

    if ( FinalCell *fc = cell->final_cell() ) {
        for( size_t n = 0; n < fc->nb_diracs(); ++n ) {
            const Dirac &dirac = fc->diracs[ n ];
            for( int l = 0; l <= level; ++l )
                path[ l ]->bounds.push( dirac.pos, dirac.weight );
        }
        return;
    }

    if ( SuperCell *sc =cell->super_cell() ) {
        for( int i = 0; i < sc->nb_sub_cells(); ++i )
            update_cell_bounds_phase_1( sc->sub_cells[ i ], path, level + 1 );
        return;
    }

    TODO;
}

template<class Pc>
void LGrid<Pc>::display_tikz( std::ostream &/*os*/, DisplayFlags /*display_flags*/ ) const {
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
void LGrid<Pc>::display_vtk( VtkOutput &vtk_output, DisplayFlags display_flags ) const {
    if ( root_cell )
        display_vtk( vtk_output, root_cell, display_flags );
}

template<class Pc>
void LGrid<Pc>::display_vtk( VtkOutput &vtk_output, BaseCell *cell, DisplayFlags display_flags ) const {
    Pt a = cell->bounds.min_pos, b = cell->bounds.max_pos;
    std::vector<Point3<TF>> pts = {
        Point3<TF>{ a[ 0 ], a[ 1 ], 0 },
        Point3<TF>{ b[ 0 ], a[ 1 ], 0 },
        Point3<TF>{ b[ 0 ], b[ 1 ], 0 },
        Point3<TF>{ a[ 0 ], b[ 1 ], 0 },
    };
    if ( dim == 2 && display_flags.weight_elevation )
        for( Point3<TF> &pt : pts )
            pt[ 2 ] = display_flags.weight_elevation * cell->bounds.get_w( { pt[ 0 ], pt[ 1 ] } );
    vtk_output.add_polygon( pts );

    if ( cell->super_cell() ) {
        const SuperCell *sc = static_cast<const SuperCell *>( cell );
        for( std::size_t i = 0; i < sc->nb_sub_cells(); ++i )
            display_vtk( vtk_output, sc->sub_cells[ i ], display_flags );
    }

    if ( cell->final_cell() ) {
        const FinalCell *sc = static_cast<const FinalCell *>( cell );
        if ( dim == 2 && display_flags.weight_elevation )
            for( std::size_t i = 0; i < sc->nb_diracs(); ++i )
                vtk_output.add_point( Point3<TF>{ sc->diracs[ i ].pos[ 0 ], sc->diracs[ i ].pos[ 1 ], display_flags.weight_elevation * sc->diracs[ i ].weight } );
        else
            for( std::size_t i = 0; i < sc->nb_diracs(); ++i )
                vtk_output.add_point( sc->diracs[ i ].pos );
    }
}

} // namespace sdot
