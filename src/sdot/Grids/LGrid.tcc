#include "../Geometry/Internal/ZCoords.h"
#include "../Support/StaticRange.h"
#include "../Support/RadixSort.h"
#include "../Support/ASSERT.h"
#include "../Support/Stat.h"
#include "../Support/Span.h"
#include "LGrid.h"
#include <unistd.h>
#include <fstream>
#include <cmath>

namespace sdot {

template<class Pc>
LGrid<Pc>::LGrid( std::size_t max_diracs_per_cell ) : max_diracs_per_cell( max_diracs_per_cell ) {
    // parameters
    nb_fcells_per_ooc_file = 100;
    max_usable_ram         = std::numeric_limits<std::size_t>::max();
    ooc_dir                = "ooc";

    // content
    reset();
}


template<class Pc>
LGrid<Pc>::~LGrid() {
}

template<class Pc>
void LGrid<Pc>::construct( const Dirac *diracs, TI nb_diracs ) {
    construct( [&]( const Cb &cb ) {
        cb( diracs, nb_diracs, true );
    } );
}

template<class Pc>
void LGrid<Pc>::construct( const std::function<void(const Cb &cb)> &f ) {
    // make a fresh start
    reset();

    // get min/max of positions + positions and weight pointers (if possible to use them)
    get_grid_dims_and_dirac_ptrs( f );

    // create the cells + first phase of bounds update
    make_the_cells( f );

    // second phase of bounds update (if necessary)
    if ( CellBounds::need_phase_1 && root_cell ) {
        Cell *path[ nb_bits_per_axis ];
        update_cell_bounds_phase_1( root_cell, path, 0 );
    }

    PN( *this );
}

template<class Pc>
void LGrid<Pc>::update_after_mod_weights() {
    if ( root_cell ) {
        // tmp storage for multi-level information
        LocalSolver local_solvers[ nb_bits_per_axis ];
        update_after_mod_weights_rec( root_cell, local_solvers, 0 );
    }

    // second phase of bounds update (if necessary)
    if ( CellBounds::need_phase_1 && root_cell ) {
        Cell *path[ nb_bits_per_axis ];
        update_cell_bounds_phase_1( root_cell, path, 0 );
    }
}

template<class Pc>
void LGrid<Pc>::update_after_mod_weights_rec( Cell *cell, LocalSolver *local_solvers, int level ) {
    local_solvers[ level ].clr();

    for( const Dirac &dirac : cell->diracs() ) {
        local_solvers[ level ].push( dirac.pos, dirac.weight );
    }

    for( Cell *sc : cell->scells() ) {
        update_after_mod_weights_rec( sc, local_solvers, level + 1 );
        local_solvers[ level ].push( local_solvers[ level + 1 ] );
    }

    for( std::size_t oc : cell->ocells() ) {
        oc += 1;
        TODO;
    }

    local_solvers[ level ].store_to( cell->bounds );
}

template<class Pc>
void LGrid<Pc>::get_grid_dims_and_dirac_ptrs( const std::function<void(const Cb &cb)> &f ) {
    using std::min;
    using std::max;

    // reset
    use_diracs_from_cb = true;

    // traversal
    TI nb_cb_calls = 0;
    min_point = + std::numeric_limits<TF>::max();
    max_point = - std::numeric_limits<TF>::max();
    f( [&]( const Dirac *diracs, TI nb_diracs, bool ptrs_survive_the_call ) {
        if ( nb_diracs == 0 )
            return;

        // TODO: in parallel
        for( std::size_t num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
            const Dirac &dirac = diracs[ num_dirac ];
            min_point = min( min_point, dirac.pos );
            max_point = max( max_point, dirac.pos );
        }

        if ( nb_cb_calls ) {
            use_diracs_from_cb = false;
        } else {
            use_diracs_from_cb = ptrs_survive_the_call;
            diracs_from_cb = diracs;
        }

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
void LGrid<Pc>::reset() {
    out_of_core_infos.clear();
    pool_scells.clear();

    first_in_mem_cell = nullptr;
    last_in_mem_cell  = nullptr;
    nooc_mem_cell     = 0;

    nb_final_cells    = 0;
    used_fcell_ram    = 0;
    used_scell_ram    = 0;
    nb_diracs_tot     = 0;
    nb_filenames      = 0;
    root_cell         = nullptr;
}

template<class Pc>
void LGrid<Pc>::make_the_cells( const std::function<void(const Cb &cb)> &f ) {
    static_assert( sizeof( TZ ) >= sizeof_zcoords, "TZ (zcoords type) is not large enough" );

    // tmp storage for multi-level information
    TmpLevelInfo level_info[ nb_bits_per_axis + 1 ];
    for( TmpLevelInfo &l : level_info )
        l.clr();

    //
    std::vector<TI> zind_indices;
    std::vector<TZ> zind_limits;
    const Dirac **zn_ptrs;
    TZ *zn_keys;
    if ( use_diracs_from_cb ) {
        // we don't need to look further
        zind_limits = { TZ( 1 ) << dim * nb_bits_per_axis };
        zind_indices = { nb_diracs_tot };

        // make znode_xxx lists
        znodes_keys.reserve( 2 * nb_diracs_tot );
        znodes_ptrs.reserve( 2 * nb_diracs_tot );

        TI nb_threads = thread_pool.nb_threads(), nb_jobs = nb_threads, offset = 0;
        thread_pool.execute( nb_jobs, [&]( TI num_job, int /*num_thread*/ ) {
            TI beg = ( num_job + 0 ) * nb_diracs_tot / nb_jobs;
            TI end = ( num_job + 1 ) * nb_diracs_tot / nb_jobs;
            for( TI num_dirac = beg; num_dirac < end; ++num_dirac ) {
                TZ zcoords = zcoords_for<TZ,nb_bits_per_axis>( diracs_from_cb[ num_dirac ].pos, min_point, inv_step_length );
                znodes_ptrs[ num_dirac ] = diracs_from_cb + num_dirac;
                znodes_keys[ num_dirac ] = zcoords;
            }
        } );

        // sort w.r.t. zcoords
        std::pair<TZ *,const Dirac **> sorted_znodes = radix_sort(
            std::make_pair( znodes_keys.data() + nb_diracs_tot, znodes_ptrs.data() + nb_diracs_tot ),
            std::make_pair( znodes_keys.data(), znodes_ptrs.data() ),
            nb_diracs_tot,
            N<dim*nb_bits_per_axis>(),
            rs_tmps
        );
        zn_keys = sorted_znodes.first;
        zn_ptrs = sorted_znodes.second;
    } else {
        make_zind_limits( zind_indices, zind_limits, f );
    }

    TZ num_in_zind_limits = 0;
    auto check_possible_read = [&]( TI index ) -> void {
        if ( nb_diracs_tot == 0 )
            return;
        if ( index >= nb_diracs_tot )
            index = nb_diracs_tot - 1;

        if ( index < zind_indices[ num_in_zind_limits ] )
            return;

        // else, go to next zone
        ++num_in_zind_limits;

        // and get the data
        TZ bo_ld = max_diracs_per_cell + zind_indices[ num_in_zind_limits ] - zind_indices[ num_in_zind_limits - 1 ];
        TZ beg_z = index > max_diracs_per_cell ? zn_keys[ index - max_diracs_per_cell ] : 0;
        TZ end_z = zind_limits[ num_in_zind_limits ];

        tmp_diracs.clear();
        tmp_diracs.reserve( bo_ld );
        znodes_keys.reserve( 2 * bo_ld );
        znodes_ptrs.reserve( 2 * bo_ld );

        TZ nb_ld = 0;
        TI beg_i = 0;
        f( [&]( const Dirac *diracs, TI nb_diracs, bool /*ptrs_survive_the_call*/ ) {
            for( std::size_t num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
                TZ zcoords = zcoords_for<TZ,nb_bits_per_axis>( diracs[ num_dirac ].pos, min_point, inv_step_length );
                if ( zcoords < beg_z )
                    ++beg_i;
                else if ( zcoords < end_z ) {
                    tmp_diracs.push_back( diracs[ num_dirac ] );
                    znodes_keys[ nb_ld ] = zcoords;
                    znodes_ptrs[ nb_ld ] = tmp_diracs.data() + nb_ld;
                    ++nb_ld;
                }
            }
        } );

        std::pair<TZ *,const Dirac **> sorted_znodes = radix_sort(
            std::make_pair( znodes_keys.data() + nb_ld, znodes_ptrs.data() + nb_ld ),
            std::make_pair( znodes_keys.data(), znodes_ptrs.data() ),
            nb_ld,
            N<dim*nb_bits_per_axis>(),
            rs_tmps
        );
        zn_keys = sorted_znodes.first - beg_i;
        zn_ptrs = sorted_znodes.second - beg_i;
    };

    // get the cells zcoords and indices (offsets in dpc_indices) + dpc_indices
    int level = 0;
    TZ prev_z = 0;
    for( TI index = max_diracs_per_cell; ; ) {
        // check availability of the dirac info
        check_possible_read( index );

        // last cell(s)
        if ( index >= nb_diracs_tot ) {
            while ( prev_z < ( TZ( 1 ) << dim * nb_bits_per_axis ) ) {
                for( ; ; ++level ) {
                    TZ m = TZ( 1 ) << dim * ( level + 1 );
                    if ( level == nb_bits_per_axis || prev_z & ( m - 1 ) ) {
                        push_cell( nb_diracs_tot, prev_z, level, level_info, index, zn_ptrs, zn_keys );
                        break;
                    }
                }
            }
            break;
        }

        // level too high ?
        for( ; ; --level ) {
            TZ m = TZ( 1 ) << dim * ( level + 1 );
            if ( zn_keys[ index ] >= prev_z + m )
                break;
            if ( level == 0 )
                break;
        }

        // look for a level before the one that will take the $max_diracs_per_cell next points or that will lead to an illegal cell
        for( ; ; ++level ) {
            TZ m = TZ( 1 ) << dim * ( level + 1 );
            if ( zn_keys[ index ] < prev_z + m || ( prev_z & ( m - 1 ) ) ) {
                push_cell( index, prev_z, level, level_info, index, zn_ptrs, zn_keys );
                break;
            }
        }
    }
}

template<class Pc>
void LGrid<Pc>::push_cell( TI l, TZ &prev_z, TI level, TmpLevelInfo *level_info, TI &index, const Dirac **zn_ptrs, TZ *zn_keys ) {
    TZ old_prev_z = prev_z;
    prev_z += TZ( 1 ) << dim * level;

    // beg/end of cells to push (indices in sorted_znodes)
    TI beg_ind_zn = l, len_ind_nz = 0;
    for( TI n = index - max_diracs_per_cell; n < l; ++n ) {
        if ( zn_keys[ n ] >= old_prev_z ) {
            beg_ind_zn = n;
            for( ; ; ++n  ) {
                if ( n == l || zn_keys[ n ] >= prev_z ) {
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
    Cell *cell = nullptr;
    if ( len_ind_nz ) {
        // get the pool
        TI nooc = nb_final_cells / nb_fcells_per_ooc_file;
        if ( out_of_core_infos.size() <= nooc )
            out_of_core_infos.resize( nooc + 1 );

        // create a new fcell
        cell = Cell::allocate( out_of_core_infos[ nooc ].pool, used_fcell_ram, len_ind_nz, 0, 0 );
        cell->end_ind_in_fcells = ++nb_final_cells;

        // register it in the linked list of the in_mem fcells
        if ( last_in_mem_cell )
            last_in_mem_cell->first_alloc_data().next = cell;
        else
            first_in_mem_cell = cell;
        last_in_mem_cell = cell;

        // store diracs indices, get bounds
        LocalSolver ls;
        ls.clr();
        for( TI i = 0; i < len_ind_nz; ++i ) {
            const Dirac &dirac = *zn_ptrs[ beg_ind_zn + i ];
            cell->dirac( i ) = dirac;

            ls.push( dirac.pos, dirac.weight );
        }

        // store the cell and the bounds
        li->scells[ li->nb_scells++ ] = cell;
        ls.store_to( cell->bounds );
        li->ls.push( ls );

        //
        while( used_fcell_ram > max_usable_ram )
            free_ooc( nooc_mem_cell++, level_info, level );
    }

    // multilevel
    for( std::size_t sl = level; ; sl++ ) {
        // coarser level ?
        if ( sl == nb_bits_per_axis ) {
            root_cell = cell;
            break;
        }

        // if the sub cells are not finished, stay in this level
        if ( li->num_scell < ( 1 << dim ) - 1 ) {
            ++li->num_scell;
            break;
        }

        // else, make a new super cell
        TmpLevelInfo *oli = li++;
        if ( oli->nb_scells + oli->nb_ocells ) {
            if ( oli->nb_scells + oli->nb_ocells > 1 ) {
                cell = Cell::allocate( pool_scells, used_scell_ram, 0, oli->nb_scells, oli->nb_ocells );
                cell->end_ind_in_fcells = nb_final_cells;
                for( std::size_t i = 0; i < oli->nb_scells; ++i ) {
                    cell->scell( i ) = oli->scells[ i ];

                    oli->scells[ i ]->first_alloc_data().num_in_parent = i;
                    oli->scells[ i ]->first_alloc_data().parent = cell;
                }
                for( std::size_t i = 0; i < oli->nb_ocells; ++i ) {
                    cell->ocell( i ) = oli->ocells[ i ];
                }

                oli->ls.store_to( cell->bounds );

                li->scells[ li->nb_scells++ ] = cell;
            } else {
                li->scells[ li->nb_scells++ ] = oli->scells[ 0 ];
            }

            li->ls.push( oli->ls );
        }

        // and reset the previous level
        oli->clr();
    }
}


template<class Pc>
LGrid<Pc>::OutOfCoreInfo::OutOfCoreInfo( OutOfCoreInfo &&that ) : filename( std::move( that.filename ) ), in_memory( that.in_memory ), saved( that.saved ), pool( std::move( that.pool ) ) {
    that.filename.clear();
}

template<class Pc>
LGrid<Pc>::OutOfCoreInfo::~OutOfCoreInfo() {
    if ( ! filename.empty() )
        unlink( filename.c_str() );
}


template<class Pc>
void LGrid<Pc>::free_ooc( TI nooc, TmpLevelInfo *level_info, TI level ) {
    OutOfCoreInfo &oi = out_of_core_infos[ nooc ];
    if ( oi.filename.empty() )
        oi.filename = va_string( "{}{}_{}_{}.bin", ooc_dir, getpid(), this, nb_filenames++ );
    oi.in_memory = false;
    oi.saved = true;

    std::ofstream fout( oi.filename.c_str() );
    fout.write( (char *)&nb_fcells_per_ooc_file, sizeof( TI ) );
    for( std::size_t n = 0; n < nb_fcells_per_ooc_file; ++n ) {
        Cell *cell = first_in_mem_cell;
        ASSERT( cell, "" );

        std::size_t wr_size = cell->size_in_bytes( /*first_alloc*/ false );
        used_fcell_ram -= cell->size_in_bytes( /*first_alloc*/ true );
        fout.write( (char *)cell, wr_size );

        if ( Cell *p = cell->first_alloc_data().parent ) {
            int np = cell->first_alloc_data().num_in_parent;
            for( int n = np + 1; n < p->nb_scells; ++n ) {
                p->scell( n )->first_alloc_data().num_in_parent = n - 1;
                p->scell( n - 1 ) = p->scell( n );
            }
            --p->nb_scells;

            p->ocell( 0 ) = cell->end_ind_in_fcells - 1;
            ++p->nb_ocells;
        } else {
            // find it in level_info
            auto find_cell = [&]() {
                for( TI l = 0; l <= level; ++l ) {
                    for( int i = 0; i < level_info[ l ].nb_scells; ++i ) {
                        if ( level_info[ l ].scells[ i ] == cell ) {
                            for( int n = i + 1; n < level_info[ l ].nb_scells; ++n )
                                level_info[ l ].scells[ i - 1 ] = level_info[ l ].scells[ i ];
                            level_info[ l ].ocells[ level_info[ l ].nb_ocells++ ] = cell->end_ind_in_fcells - 1;
                            --level_info[ l ].nb_scells;
                            return true;
                        }
                    }
                }
                return false;
            };
            if ( ! find_cell() )
                ERROR( "cell is registerd nowhere" );
        }

        first_in_mem_cell = first_in_mem_cell->first_alloc_data().next;
    }

    // mettre offset dans scell
    oi.pool.free();
}

template<class Pc>
void LGrid<Pc>::read_ooc_for( TI off ) {
    TI noi = off / nb_fcells_per_ooc_file;
    OutOfCoreInfo &oi = out_of_core_infos[ noi ];
    if ( oi.in_memory )
        return;
    ASSERT( oi.saved, "" );

    // read the raw data
    std::ifstream fin( oi.filename.c_str(), std::ios::binary | std::ios::ate );
    std::size_t tot_size = fin.tellg();
    char *alloc = oi.pool.allocate( tot_size );
    fin.seekg( 0 );
    fin.read( alloc, tot_size );

    // nb cells in the file
    TI nb_cells = *reinterpret_cast<const TI *>( alloc );
    alloc += sizeof( TI );

    // offsets
    std::vector<Cell *> cells( nb_cells );
    for( TI i = 0; i < nb_cells; ++i ) {
        Cell *cell = reinterpret_cast<Cell *>( alloc );
        std::size_t s = cell->size_in_bytes( false );
        cells[ i ] = cell;
        alloc += s;
    }

    // replace offsets
    replace_ooc_offsets_py_ptrs( noi * nb_fcells_per_ooc_file, cells, root_cell );
}

template<class Pc>
void LGrid<Pc>::replace_ooc_offsets_py_ptrs( TI beg_cell, const std::vector<Cell *> &cells, Cell *cell ) {
    if ( cell->nb_diracs )
        return;

    TI end_cell = beg_cell + cells.size();
    if ( int n = cell->nb_ocells ) {
        while( n-- ) {
            std::size_t off = cell->ocell( n );
            if ( off >= beg_cell && off < end_cell ) {
                if ( n )
                    cell->ocell( n - 1 ) = cell->ocell( 0 );
                cell->scell( cell->nb_scells++ ) = cells[ off - beg_cell ];
                --cell->nb_ocells;
            }
        }

        Span<Cell *> sp = cell->scells();
        std::sort( const_cast<Cell **>( sp.begin() ), const_cast<Cell **>( sp.end() ), []( Cell *a, Cell *b ) {
            return a->end_ind_in_fcells < b->end_ind_in_fcells;
        } );
    }

    for( int i = 0; i < cell->nb_scells; ++i ) {
        Cell *sc = cell->scell( i );
        if ( sc->end_ind_in_fcells <= beg_cell )
            continue;

        replace_ooc_offsets_py_ptrs( beg_cell, cells, sc );

        if ( sc->end_ind_in_fcells <= beg_cell )
            break;
    }
}

template<class Pc>
void LGrid<Pc>::make_zind_limits( std::vector<TI> &zind_indices, std::vector<TZ> &zind_limits, const std::function<void(const Cb &)> &f ) {
    // => get nb items / cell in a regular grid
    constexpr int nb_bits_items = 26, shift = dim * nb_bits_per_axis - nb_bits_items;
    std::vector<TI> nb_items( TZ( 1 ) << nb_bits_items );
    f( [&]( const Dirac *diracs, TI nb_diracs, bool /*ptrs_survive_the_call*/ ) {
        for( std::size_t num_dirac = 0; num_dirac < nb_diracs; ++num_dirac ) {
            TZ zcoords = zcoords_for<TZ,nb_bits_per_axis>( diracs[ num_dirac ].pos, min_point, inv_step_length );
            TZ ind = zcoords >> shift;
            ++nb_items[ ind ];
        }
    } );

    //
    zind_indices = { 0 };
    zind_limits = { 0 };
    for( TZ n = 0, t = 0; n < nb_items.size(); ) {
        TI acc = 0;
        for( ; n < nb_items.size(); ++n ) {
            TI tmp = acc + nb_items[ n ];
            if ( tmp * sizeof( Dirac ) > max_usable_ram )
                break;
            acc = tmp;
        }

        t += acc;
        zind_indices.push_back( t );
        zind_limits.push_back( TZ( n ) << shift );
    }
}


template<class Pc> template<int avoid_n0,int flags>
void LGrid<Pc>::cut_lc( CP &lc, Point<TF,2> c0, TF w0, Cell *dell, N<avoid_n0>, TI n0, N<flags> ) const {
    struct alignas( 64 ) Cut {
        TF     dx[ 128 ];
        TF     dy[ 128 ];
        TF     ps[ 128 ];
        Dirac *id[ 128 ];
    };
    Cut cut;

    //    #ifdef __AVX512F__
    //    TI n1 = 0, nb_cuts = dell->nb_diracs();
    //    for( ; n1 + 8 <= nb_cuts; n1 += 8 ) {
    //        __m512d cx = _mm512_set1_pd( c0.x );
    //        __m512d cy = _mm512_set1_pd( c0.y );
    //        __m512i i1 = _mm512_loadu_si512( dell->dirac_indices + n1 );
    //        __m512d vx = _mm512_sub_pd( _mm512_i64gather_pd( i1, positions[ 0 ], 8 ), cx );
    //        __m512d vy = _mm512_sub_pd( _mm512_i64gather_pd( i1, positions[ 1 ], 8 ), cy );
    //        __m512d v2 = _mm512_add_pd( _mm512_mul_pd( vx, vx ), _mm512_mul_pd( vy, vy ) );
    //        __m512d ps = _mm512_add_pd( _mm512_add_pd( _mm512_mul_pd( cx, vx ), _mm512_mul_pd( cy, vy ) ),
    //                                    _mm512_set1_pd( 0.5 ) * ( flags & homogeneous_weights ? v2 : _mm512_add_pd( v2, _mm512_set1_pd( w0 ) ) - _mm512_i64gather_pd( i1, weights, 8 ) ) );
    //        _mm512_store_pd( cut.dx + n1, vx );
    //        _mm512_store_pd( cut.dy + n1, vy );
    //        _mm512_store_pd( cut.ps + n1, ps );
    //        _mm512_store_epi64( cut.id + n1, i1 );
    //    }
    //    for( ; n1 < nb_cuts; ++n1 ) {
    //        TI i1 = dell->dirac_indices[ n1 ];
    //        Pt dc = pt( positions, i1 ) - c0;
    //        TF w1 = weights[ i1 ];
    //        cut.dx[ n1 ] = dc.x;
    //        cut.dy[ n1 ] = dc.y;
    //        cut.id[ n1 ] = i1;
    //        cut.ps[ n1 ] = dot( c0, dc ) + TF( 0.5 ) * ( flags & homogeneous_weights ? norm_2_p2( dc ) : norm_2_p2( dc ) + w0 - w1 );
    //    }

    //    if ( avoid_n0 ) {
    //        --nb_cuts;
    //        cut.dx[ n0 ] = cut.dx[ nb_cuts ];
    //        cut.dy[ n0 ] = cut.dy[ nb_cuts ];
    //        cut.id[ n0 ] = cut.id[ nb_cuts ];
    //        cut.ps[ n0 ] = cut.ps[ nb_cuts ];
    //    }

    //    #else
    TI nb_cuts = 0;
    for( std::size_t n1 = 0; n1 < dell->nb_diracs; ++n1 ) {
        if ( avoid_n0 && n1 == n0 )
            continue;
        Dirac &d1 = dell->dirac( n1 );
        Pt c1 = d1.pos;
        TF dw = flags & homogeneous_weights ? 0 : d1.weight - w0;
        cut.dx[ nb_cuts ] = c1.x - c0.x;
        cut.dy[ nb_cuts ] = c1.y - c0.y;
        cut.id[ nb_cuts ] = &d1;
        cut.ps[ nb_cuts ] = TF( 0.5 ) * ( norm_2_p2( c1 ) - norm_2_p2( c0 ) - dw );
        ++nb_cuts;
    }
    //    #endif

    // do the cuts
    lc.plane_cut( { cut.dx, cut.dy }, cut.ps, cut.id, nb_cuts );
}

template<class Pc> template<int avoid_n0,int flags>
void LGrid<Pc>::cut_lc( CP &lc, Point<TF,3> c0, TF w0, Cell *dell, N<avoid_n0>, TI n0, N<flags> ) const {
    struct alignas( 64 ) Cut {
        TF     dx[ 128 ];
        TF     dy[ 128 ];
        TF     dz[ 128 ];
        TF     ps[ 128 ];
        Dirac *id[ 128 ];
    };
    Cut cut;

    //    #ifdef __AVX512F__
    //    TI n1 = 0, nb_cuts = dell->nb_diracs();
    //    for( ; n1 + 8 <= nb_cuts; n1 += 8 ) {
    //        __m512d cx = _mm512_set1_pd( c0.x );
    //        __m512d cy = _mm512_set1_pd( c0.y );
    //        __m512i i1 = _mm512_loadu_si512( dell->dirac_indices + n1 );
    //        __m512d vx = _mm512_sub_pd( _mm512_i64gather_pd( i1, positions[ 0 ], 8 ), cx );
    //        __m512d vy = _mm512_sub_pd( _mm512_i64gather_pd( i1, positions[ 1 ], 8 ), cy );
    //        TODO;
    //        __m512d v2 = _mm512_add_pd( _mm512_mul_pd( vx, vx ), _mm512_mul_pd( vy, vy ) );
    //        __m512d ps = _mm512_add_pd( _mm512_add_pd( _mm512_mul_pd( cx, vx ), _mm512_mul_pd( cy, vy ) ),
    //                                    _mm512_set1_pd( 0.5 ) * ( flags & homogeneous_weights ? v2 : _mm512_add_pd( v2, _mm512_set1_pd( w0 ) ) - _mm512_i64gather_pd( i1, weights, 8 ) ) );
    //        _mm512_store_pd( cut.dx + n1, vx );
    //        _mm512_store_pd( cut.dy + n1, vy );
    //        _mm512_store_pd( cut.ps + n1, ps );
    //        _mm512_store_epi64( cut.id + n1, i1 );
    //    }
    //    for( ; n1 < nb_cuts; ++n1 ) {
    //        TI i1 = dell->dirac_indices[ n1 ];
    //        Pt dc = pt( positions, i1 ) - c0;
    //        TF w1 = weights[ i1 ];
    //        cut.dx[ n1 ] = dc.x;
    //        cut.dy[ n1 ] = dc.y;
    //        cut.id[ n1 ] = i1;
    //        cut.ps[ n1 ] = dot( c0, dc ) + TF( 0.5 ) * ( flags & homogeneous_weights ? norm_2_p2( dc ) : norm_2_p2( dc ) + w0 - w1 );
    //    }

    //    if ( avoid_n0 ) {
    //        --nb_cuts;
    //        cut.dx[ n0 ] = cut.dx[ nb_cuts ];
    //        cut.dy[ n0 ] = cut.dy[ nb_cuts ];
    //        cut.id[ n0 ] = cut.id[ nb_cuts ];
    //        cut.ps[ n0 ] = cut.ps[ nb_cuts ];
    //    }

    //    #else
    TI nb_cuts = 0;
    for( int n1 = 0; n1 < dell->nb_diracs; ++n1 ) {
        if ( avoid_n0 && n1 == n0 )
            continue;
        Dirac &d1 = dell->dirac( n1 );
        Pt c1 = d1.pos;
        TF dw = flags & homogeneous_weights ? 0 : d1.weight - w0;
        cut.dx[ nb_cuts ] = c1.x - c0.x;
        cut.dy[ nb_cuts ] = c1.y - c0.y;
        cut.dz[ nb_cuts ] = c1.z - c0.z;
        cut.id[ nb_cuts ] = &d1;
        cut.ps[ nb_cuts ] = TF( 0.5 ) * ( norm_2_p2( c1 ) - norm_2_p2( c0 ) - dw );
        ++nb_cuts;
    }
    //    #endif

    // do the cuts
    // TODO: integration of diracs in lc.plane_cut
    lc.plane_cut( { cut.dx, cut.dy, cut.dz }, cut.ps, cut.id, nb_cuts );
}

template<class Pc> template<int flags,class SLC>
void LGrid<Pc>::make_lcs_from( const std::function<void( CP &, Dirac &dirac, int num_thread )> &cb, std::priority_queue<LGrid::Msi> &base_queue, std::priority_queue<LGrid::Msi> &queue, LGrid::CP &lc, Cell *cell, const LGrid::CpAndNum *path, int path_len, int num_thread, N<flags>, const SLC &starting_lc ) const {
    // helper to add a cell in the queue
    auto append_msi = [&]( std::priority_queue<Msi> &queue, Cell *dell, Pt cell_center ) {
        Pt dell_center = 0.5 * ( dell->bounds.min_pos + dell->bounds.max_pos );
        queue.push( Msi{ dell_center, dell, norm_2( dell_center - cell_center ) } );
    };

    // fill a first queue
    base_queue = {};
    const Pt cell_center = 0.5 * ( cell->bounds.min_pos + cell->bounds.max_pos );
    for( int num_in_path = 0; num_in_path < path_len; ++num_in_path )
        for( int i = 0; i < path[ num_in_path ].cell->nb_scells; ++i )
            if ( i != path[ num_in_path ].num )
                append_msi( base_queue, path[ num_in_path ].cell->scell( i ), cell_center );

    // for each dirac
    for( std::size_t n0 = 0; n0 < cell->nb_diracs; ++n0 ) {
        Dirac &d0 = cell->dirac( n0 );
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
            if ( msi.cell->nb_diracs ) {
                cut_lc( lc, c0, w0, msi.cell, N<0>(), 0, N<flags>() );
                continue;
            }

            // else, add sub_cells in the queue
            for( int i = 0; i < msi.cell->nb_scells; ++i )
                append_msi( queue, msi.cell->scell( i ), c0 );

            //
            for( int i = 0; i < msi.cell->nb_ocells; ++i )
                TODO;
        }

        //
        cb( lc, d0, num_thread );
    }
}

template<class Pc> template<class SLC>
int LGrid<Pc>::for_each_laguerre_cell( const std::function<void( CP &, Dirac &dirac, int num_thread )> &cb, const SLC &starting_lc, TraversalFlags traversal_flags ) const {
    constexpr int flags = 0;
    int err;

    // parallel traversal of the cells
    int nb_threads = thread_pool.nb_threads(), nb_jobs = nb_threads;
    thread_pool.execute( nb_jobs, [&]( std::size_t num_job, int num_thread ) {
        TI beg_cell = ( num_job + 0 ) * nb_final_cells / nb_jobs;
        TI end_cell = ( num_job + 1 ) * nb_final_cells / nb_jobs;
        std::priority_queue<Msi> base_queue, queue;
        CP lc;

        for_each_final_cell_mono_thr( [&]( Cell &cell, CpAndNum *path, TI path_len ) {
            make_lcs_from( cb, base_queue, queue, lc, &cell, path, path_len, num_thread, N<flags>(), starting_lc );
        }, beg_cell, end_cell, root_cell );
    } );

    //
    if ( traversal_flags.mod_weights )
        const_cast<LGrid *>( this )->update_after_mod_weights();

    return err;
}


template<class Pc>
void LGrid<Pc>::for_each_final_cell_mono_thr( const std::function<void( Cell &cell, CpAndNum *path, TI path_len )> &f, TI beg_cell, TI end_cell, Cell *root_cell ) const {
    if ( ! root_cell )
        return;

    // path to super cells, starting from `beg_cell`
    TI path_len = 0;
    CpAndNum path[ nb_bits_per_axis ];
    for( Cell *cell = root_cell; ; ++path_len ) {
        if ( cell->nb_diracs )
            break;

        for( int i = 0; ; ++i ) {
            if ( i == cell->nb_scells )
                TODO;
            Cell *suc = cell->scell( i );
            if ( suc->end_ind_in_fcells > beg_cell ) {
                path[ path_len ].cell = cell;
                path[ path_len ].num = i;
                cell = suc;
                break;
            }
        }
    }

    //
    if ( path_len == 0 ) {
        f( *root_cell, path, path_len );
        return;
    }

    // up to end_cell
    TI end_indc = end_cell + 1;
    while ( true ) {
        Cell *fcell = path[ path_len - 1 ].cell->scell( path[ path_len - 1 ].num );
        if ( fcell->end_ind_in_fcells == end_indc )
            return;

        // call
        f( *fcell, path, path_len );

        // new scell/num pair
        while ( ++path[ path_len - 1 ].num == path[ path_len - 1 ].cell->nb_scells )
            if ( --path_len == 0 )
                return;

        // go deeper, to the first final cell
        while ( true ) {
            Cell *tspc = path[ path_len - 1 ].cell->scell( path[ path_len - 1 ].num );
            if ( tspc->nb_diracs )
                break;

            path[ path_len ].cell = tspc;
            path[ path_len ].num = 0;
            ++path_len;
        }
    }
}


template<class Pc>
void LGrid<Pc>::for_each_final_cell( const std::function<void( Cell &cell, int num_thread )> &f, TraversalFlags traversal_flags ) const {
    // parallel traversal of the cells
    int nb_threads = thread_pool.nb_threads(), nb_jobs = nb_threads;
    thread_pool.execute( nb_jobs, [&]( std::size_t num_job, int num_thread ) {
        TI beg_cell = ( num_job + 0 ) * nb_final_cells / nb_jobs;
        TI end_cell = ( num_job + 1 ) * nb_final_cells / nb_jobs;
        for_each_final_cell_mono_thr( [&]( Cell &cell, CpAndNum */*path*/, TI /*path_len*/ ) {
            f( cell, num_thread );
        }, beg_cell, end_cell );
    } );

    //
    if ( traversal_flags.mod_weights )
        update_after_mod_weights();
}

template<class Pc>
void LGrid<Pc>::for_each_dirac( const std::function<void( Dirac &, int )> &f, TraversalFlags traversal_flags ) const {
    for_each_final_cell( [&]( Cell &cell, int num_thread ) {
        for( std::size_t i = 0; i < cell.nb_diracs; ++i )
            f( cell.dirac( i ), num_thread );
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
    if ( root_cell )
        root_cell->write_to_stream( os, {} );
    else
        os << "null";
}



template<class Pc>
void LGrid<Pc>::update_cell_bounds_phase_1( Cell *cell, Cell **path, int level ) {
    path[ level ] = cell;

    for( const Dirac &dirac : cell->diracs() )
        for( int l = 0; l <= level; ++l )
            path[ l ]->bounds.push( dirac.pos, dirac.weight );

    while( cell->nb_ocells )
        read_ooc_for( cell->ocell( 0 ) );

    for( Cell *sc : cell->scells() )
        update_cell_bounds_phase_1( sc, path, level + 1 );

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
    if ( display_flags.display_boxes && root_cell )
        display_vtk( vtk_output, root_cell, display_flags );

    if ( display_flags.display_cells ) {
        std::mutex m;
        typename CP::Box ic( { TF( 0 ), TF( 1 ) } );
        for_each_laguerre_cell( [&]( auto &cp, auto &d, int ) {
            m.lock();
            cp.display_vtk( vtk_output, d.values() );
            m.unlock();
        }, ic );
    }
}

template<class Pc>
void LGrid<Pc>::display_vtk( VtkOutput &vtk_output, Cell *cell, DisplayFlags display_flags ) const {
    // bounds
    Pt a = cell->bounds.min_pos, b = cell->bounds.max_pos;
    std::vector<Point3<TF>> pts = {
        Point3<TF>{ a[ 0 ], a[ 1 ], 0 },
        Point3<TF>{ b[ 0 ], a[ 1 ], 0 },
        Point3<TF>{ b[ 0 ], b[ 1 ], 0 },
        Point3<TF>{ a[ 0 ], b[ 1 ], 0 },
    };
    if ( dim == 2 && display_flags.weight_elevation )
        for( Point3<TF> &pt : pts ) {
            Pt p;
            p.x = pt.x;
            p.y = pt.y;
            pt[ 2 ] = display_flags.weight_elevation * cell->bounds.get_w( p );
        }
    vtk_output.add_polygon( pts );

    // points
    if ( dim == 2 && display_flags.weight_elevation )
        for( const Dirac &d : cell->diracs() )
            vtk_output.add_point( Point3<TF>{ d.pos[ 0 ], d.pos[ 1 ], display_flags.weight_elevation * d.weight } );
    else
        for( const Dirac &d : cell->diracs() )
            vtk_output.add_point( d.pos );

    // sub cells
    for( Cell *sc : cell->scells() )
        display_vtk( vtk_output, sc, display_flags );
}

} // namespace sdot

