#include "Geometry/Internal/ZCoords.h"
#include "Support/StaticRange.h"
#include "Support/RadixSort.h"
#include "Support/ASSERT.h"
#include "Support/Stat.h"
#include "Support/Span.h"
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
    struct CpAndNum { const SuperCell *cell; TI num; };

    //    //
    //    auto cell_cut = [&]( const Cell *cell, const Cell *dell ) {
    //        if ( cell == cells.data() )
    //            P( dell - cells.data() );
    //    };

    //
    int err;
    auto make_lc_from = [&]( const FinalCell *cell, const CpAndNum *path, TI path_len )  {
        struct Msi {
            bool            operator<( const Msi &that ) const { return dist > that.dist; }
            Pt              center;
            const BaseCell *cell;
            TF              dist;
        };

        const Pt cell_center = 0.5 * ( cell->min_pos + cell->max_pos );
        auto append_msi = [&]( std::priority_queue<Msi> &queue, const BaseCell *dell ) {
            Pt dell_center = 0.5 * ( dell->min_pos + dell->max_pos );
            queue.push( Msi{ dell_center, dell, norm_2( dell_center - cell_center ) } );
        };


        // fill a first queue
        TF size = grid_length;
        std::priority_queue<Msi> queue;
        for( std::size_t num_in_path = 0; num_in_path < path_len; ++num_in_path )
            for( std::size_t i = 0; i < path[ num_in_path ].cell->nb_sub_cells(); ++i )
                if ( i != path[ num_in_path ].num )
                    append_msi( queue, path[ num_in_path ].cell->sub_cells[ i ] );

        while ( ! queue.empty() ) {
            Msi msi = queue.top();
            queue.pop();

            // if final cell, do the cuts and continue the loop
            if ( msi.cell->final_cell() ) {
                continue;
            }

            // if not potential cut, we don't go further
            const SuperCell *spc = static_cast<const SuperCell *>( msi.cell );

            // else, add sub_cells in the queue
            for( std::size_t i = 0; i < spc->nb_sub_cells(); ++i )
                append_msi( queue, spc->sub_cells[ i ] );
        }
    };

    if ( ! root_cell )
        return err;
    if ( root_cell->final_cell() ) {
        const FinalCell *cell =static_cast<const FinalCell *>( root_cell );
        make_lc_from( cell, nullptr, 0 );
        return err;
    }

    // parallel traversal of the cells
    int nb_threads = thread_pool.nb_threads(), nb_jobs = nb_threads;
    thread_pool.execute( nb_jobs, [&]( std::size_t num_job, int /*num_thread*/ ) {
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
        while ( true ) {
            const BaseCell  *lbce = path[ path_len - 1 ].cell->sub_cells[ path[ path_len - 1 ].num ];
            const FinalCell *cell = static_cast<const FinalCell *>( lbce );
            if ( cell->end_ind_in_fcells == end_indc )
                return;

            // current cell
            make_lc_from( cell, path, path_len );

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
        void      reset() {
            num_sub_cell = 0;
            nb_sub_cells = 0;
            max_weight   = - std::numeric_limits<TF>::max();
            min_pos      = + std::numeric_limits<TF>::max();
            max_pos      = - std::numeric_limits<TF>::max();
        }
        TI        num_sub_cell;          ///<
        TI        nb_sub_cells;          ///<
        BaseCell *sub_cells[ 1 << dim ]; ///<

        TF        max_weight;            ///<
        Pt        min_pos;               ///<
        Pt        max_pos;               ///<
    };

    LevelInfo level_info[ nb_bits_per_axis + 1 ];
    for( LevelInfo &l : level_info )
        l.reset();

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

            // prepare a new cell, register it in level_info
            LevelInfo *li = level_info + level;
            BaseCell *cell = nullptr;
            if ( len_ind_nz ) {
                cell = reinterpret_cast<BaseCell *>( mem_pool.allocate( sizeof( BaseCell ) + len_ind_nz * sizeof( TI ) ) );
                FinalCell *fcell = static_cast<FinalCell *>( cell );
                cell->end_ind_in_fcells = ++nb_final_cells;
                cell->nb_sub_items = len_ind_nz;

                // store diracs indices and get bounds
                TF max_weight = - std::numeric_limits<TF>::max();
                Pt min_pos = + std::numeric_limits<TF>::max();
                Pt max_pos = - std::numeric_limits<TF>::max();
                for( TI i = 0; i < len_ind_nz; ++i ) {
                    TI ind = sorted_znodes.second[ beg_ind_zn + i ];
                    fcell->dirac_indices[ i ] = ind;

                    max_weight = max( max_weight, weights[ ind ] );
                    max_pos = max( max_pos, pt( positions, ind ) );
                    min_pos = min( min_pos, pt( positions, ind ) );
                }

                //
                cell->max_weight = max_weight;
                cell->min_pos = min_pos;
                cell->max_pos = max_pos;

                //
                li->sub_cells[ li->nb_sub_cells++ ] = cell;

                li->max_weight = max( li->max_weight, max_weight );
                li->max_pos = max( li->max_pos, max_pos );
                li->min_pos = min( li->min_pos, min_pos );
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
                        cell->max_weight = oli->max_weight;
                        cell->min_pos = oli->min_pos;
                        cell->max_pos = oli->max_pos;
                    } else {
                        cell = oli->sub_cells[ 0 ];
                    }

                    //
                    li->sub_cells[ li->nb_sub_cells++ ] = cell;

                    li->max_weight = max( li->max_weight, oli->max_weight );
                    li->max_pos = max( li->max_pos, oli->max_pos );
                    li->min_pos = min( li->min_pos, oli->min_pos );
                }

                // and reset the previous level
                oli->reset();
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
    Pt a = cell->min_pos, b = cell->max_pos;

    vtk_output.add_polygon( {
        Point3<TF>{ a[ 0 ], a[ 1 ], 0 },
        Point3<TF>{ b[ 0 ], a[ 1 ], 0 },
        Point3<TF>{ b[ 0 ], b[ 1 ], 0 },
        Point3<TF>{ a[ 0 ], b[ 1 ], 0 },
        Point3<TF>{ a[ 0 ], a[ 1 ], 0 },
    } );

    if ( cell->super_cell() ) {
        const SuperCell *sc = static_cast<const SuperCell *>( cell );
        for( std::size_t i = 0; i < sc->nb_sub_cells(); ++i )
            display( vtk_output, positions, weights, sc->sub_cells[ i ], disp_weights );
    }

    if ( cell->final_cell() ) {
        const FinalCell *sc = static_cast<const FinalCell *>( cell );
        for( std::size_t i = 0; i < sc->nb_diracs(); ++i )
            vtk_output.add_point( pt( positions, sc->dirac_indices[ i ] ) );
    }

    //    auto disp_cell = [&]( Pt p, TF s, TF mw ) {
    //        if ( disp_weights ) {
    //            if ( mw == - std::numeric_limits<TF>::max() )
    //                return;
    //        }

    //        TF a = 0, b = s;
    //        switch ( dim ) {
    //        case 2: {
    //            TF wl[] = { 0, 0, 0, 0 };
    //            if ( disp_weights ) {
    //                for( std::size_t i = 0; i < 4; ++i )
    //                    wl[ i ] = mw;
    //            }
    //            vtk_output.add_polygon( {
    //                Point3<TF>{ p[ 0 ] + a, p[ 1 ] + a, wl[ 0 ] },
    //                Point3<TF>{ p[ 0 ] + b, p[ 1 ] + a, wl[ 1 ] },
    //                Point3<TF>{ p[ 0 ] + b, p[ 1 ] + b, wl[ 2 ] },
    //                Point3<TF>{ p[ 0 ] + a, p[ 1 ] + b, wl[ 3 ] },
    //                Point3<TF>{ p[ 0 ] + a, p[ 1 ] + a, wl[ 0 ] },
    //            } );
    //            break;
    //        }
    //        case 3:
    //            TODO;
    //            break;
    //        default:
    //            TODO;
    //        }
    //    };

    //    for( TI num_cell = 0; num_cell + 1 < cells.size(); ++num_cell ) {
    //        // cell
    //        Pt pos = cells[ num_cell ].pos;
    //        TF size = cells[ num_cell ].size;
    //        disp_cell( pos, size, cells[ num_cell ].max_weight );

    //        // parent ones
    //        for( TI off_pce = cells[ num_cell + 0 ].msi_offset; size *= 2, off_pce < cells[ num_cell + 1 ].msi_offset; ++off_pce )
    //            disp_cell( pos, size, msi_infos[ off_pce ].max_weight );
    //    }
}

} // namespace sdot
