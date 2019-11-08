#include "Geometry/Internal/ZCoords.h"
#include "Support/StaticRange.h"
#include "Support/RadixSort.h"
#include "Support/Stat.h"
#include "Support/Span.h"
#include "LGrid.h"
#include <cmath>
#include <set>

namespace sdot {

template<class Pc>
LGrid<Pc>::LGrid( std::size_t max_diracs_per_cell ) : max_diracs_per_cell( max_diracs_per_cell ) {
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
    return 0;
}

template<class Pc>
void LGrid<Pc>::write_to_stream( std::ostream &os ) const {
    for( std::size_t i = 0; i + 1 < cells.size(); ++i ) {
        const Cell &cell = cells[ i ];
        os << "cell num=" << i;
        if ( cells[ i ].dpc_offset != cells[ i + 1 ].dpc_offset )
            os << " max_weight=" << cell.max_weight;
        for( std::size_t j = cell.msi_offset; j < cells[ i + 1 ].msi_offset; ++j ) {
            os << "\n  ";
            const MsiInfo &msi = msi_infos[ j ];
            for( std::size_t k = 0; k < 3; ++k )
                os << " " << msi.cell_indices[ k ];
            os << "  max_weight=" << msi.max_weight;
        }
        os << "\n";
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
    znodes_keys.resize( nb_diracs );
    znodes_inds.resize( nb_diracs );
    make_znodes<nb_bits_per_axis>( znodes_keys.data(), znodes_inds.data(), positions, nb_diracs, min_point, inv_step_length );

    // sorting w.r.t. zcoords
    znodes_keys.reserve( 2 * znodes_keys.size() );
    znodes_inds.reserve( 2 * znodes_inds.size() );
    std::pair<TZ *,TI *> sorted_znodes = radix_sort(
        std::make_pair( znodes_keys.data() + znodes_keys.size(), znodes_inds.data() + znodes_inds.size() ),
        std::make_pair( znodes_keys.data(), znodes_inds.data() ),
        znodes_keys.size(),
        N<dim*nb_bits_per_axis>(),
        rs_tmps
    );

    // get the cells zcoords and indices (offsets in dpc_indices) + dpc_indices
    struct LevelInfo {
        TI  num_msi          = 0;
        TF  max_weight;
        int num_cell_indices = 2;
    };
    LevelInfo level_info[ nb_bits_per_axis ];

    int level = 0;
    TZ prev_z = 0;
    cells.resize( 0 );
    cells.reserve( znodes_keys.size() );
    dpc_indices.resize( 0 );
    dpc_indices.reserve( znodes_keys.size() );
    msi_infos.resize( 0 );
    msi_infos.reserve( znodes_keys.size() );
    for( TI index = max_diracs_per_cell; ; ) {
        auto push_cell = [&]( TI l ) {
            // new cell
            Cell cell;
            cell.zcoords = prev_z;
            cell.msi_offset = msi_infos.size();
            cell.dpc_offset = dpc_indices.size();

            // push diracs indices
            TF max_weight = - std::numeric_limits<TF>::max();
            TZ new_prev_z = prev_z + ( TZ( 1 ) << dim * level );
            for( TI n = index - max_diracs_per_cell; n < l; ++n ) {
                if ( sorted_znodes.first[ n ] >= prev_z && sorted_znodes.first[ n ] < new_prev_z ) {
                    TI ind = sorted_znodes.second[ n ];
                    dpc_indices.push_back( ind );
                    ++index;

                    max_weight = max( max_weight, weights[ ind ] );
                }
            }

            // msi_info
            auto make_msi_info = [&]( std::size_t sl, auto last_level ) {
                LevelInfo &li = level_info[ sl ];
                int ci = ++li.num_cell_indices;
                if ( ci < 3 ) {
                    MsiInfo &msi = msi_infos[ li.num_msi ];
                    msi.cell_indices[ ci ] = cells.size();

                    if ( last_level ) {
                        li.max_weight = max( li.max_weight, max_weight );
                    }

                    if ( ci == 2 ) {
                        msi.max_weight = li.max_weight;

                        if ( sl ) {
                            TF &pm = level_info[ sl - 1 ].max_weight;
                            pm = max( pm, li.max_weight );
                        }
                    }

                    return true;
                }

                // prepapre a new cell set
                li.num_cell_indices = -1;
                li.max_weight = max_weight;
                li.num_msi = msi_infos.size();

                MsiInfo new_msi;
                msi_infos.push_back( new_msi );

                return false;
            };
            if ( std::size_t sl = nb_bits_per_axis - level )
                if ( ! make_msi_info( sl, N<1>() ) )
                    while( --sl )
                        if ( make_msi_info( sl, N<0>() ) )
                            break;

            // store the cell (must be done after weight update, and msi_info which uses cells.size())
            cell.max_weight = max_weight;
            cells.push_back( cell );

            // update prev_z
            prev_z = new_prev_z;
        };

        // last cell(s)
        if ( index >= znodes_keys.size() ) {
            while ( prev_z < ( TZ( 1 ) << dim * nb_bits_per_axis ) ) {
                for( ; ; ++level ) {
                    TZ m = TZ( 1 ) << dim * ( level + 1 );
                    if ( level == nb_bits_per_axis || prev_z & ( m - 1 ) ) {
                        push_cell( znodes_keys.size() );
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

    // add an ending cell
    Cell cell;
    cell.zcoords = TZ( 1 ) << dim * nb_bits_per_axis;
    cell.dpc_offset = dpc_indices.size();
    cell.msi_offset = msi_infos.size();
    cells.push_back( cell );

    // convert zcoords to cartesian coords
    TI nb_jobs = thread_pool.nb_threads();
    thread_pool.execute( nb_jobs, [&]( TI num_job, int ) {
        TI beg = ( num_job + 0 ) * ( cells.size() - 1 ) / nb_jobs;
        TI end = ( num_job + 1 ) * ( cells.size() - 1 ) / nb_jobs;
        for( TI num_cell = beg; num_cell < end; ++num_cell ) {
            TZ zcoords = cells[ num_cell + 0 ].zcoords;
            TZ acoords = cells[ num_cell + 1 ].zcoords;

            Cell &c = cells[ num_cell ];
            c.size = step_length * round( pow( acoords - zcoords, 1.0 / dim ) );

            StaticRange<dim>::for_each( [&]( auto d ) {
                c.pos[ d ] = TF( 0 );
                StaticRange<nb_bits_per_axis>::for_each( [&]( auto i ) {
                    TZ p = zcoords & ( TZ( 1 ) << ( dim * i + d ) );
                    c.pos[ d ] += p >> ( ( dim - 1 ) * i + d );
                } );
                c.pos[ d ] = min_point[ d ] + step_length * c.pos[ d ];
            } );
        }
    } );

    cells.back().size = 0;
    cells.back().pos = max_point;
}


template<class Pc>
void LGrid<Pc>::display_tikz( std::ostream &os, TF scale ) const {
    for( TI num_cell = 0; num_cell < cells.size() - 1; ++num_cell ) {
        Pt p;
        for( int d = 0; d < dim; ++d )
            p[ d ] = cells[ num_cell ].pos[ d ];

        TF a = 0, b = cells[ num_cell ].size;
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
void LGrid<Pc>::display( VtkOutput &vtk_output ) const {
    for( TI num_cell = 0; num_cell < cells.size() - 1; ++num_cell ) {
        Pt p;
        for( int d = 0; d < dim; ++d )
            p[ d ] = cells[ num_cell ].pos[ d ];

        TF a = 0, b = cells[ num_cell ].size;
        switch ( dim ) {
        case 2:
            vtk_output.add_polygon( {
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
typename LGrid<Pc>::TZ LGrid<Pc>::ng_zcoord( TZ zcoords, TZ off, N<axis> ) const {
    using Zzoa = typename ZCoords<TZ,dim,nb_bits_per_axis>::template _ZcoordsZerosOnAxis<axis>;
    TZ ff0 = Zzoa::value;
    TZ res = ( ( zcoords | ff0 ) + off ) & ~ ff0;
    return res | ( zcoords & ff0 );
}

} // namespace sdot
