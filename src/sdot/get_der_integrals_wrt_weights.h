#pragma once

#include "Integration/SpaceFunctions/Constant.h"
#include "Integration/FunctionEnum.h"
#include "Support/ThreadPool.h"
#include "Support/ASSERT.h"
#include "Support/TODO.h"
#include <algorithm>
#include <vector>

namespace sdot {

/**
   We assume that grid has already been initialized by diracs
*/
template<class TI,class TF,class Grid,class Bounds,class Pt,class Func>
int get_der_integrals_wrt_weights( std::vector<TI> &m_offsets, std::vector<TI> &m_columns, std::vector<TF> &m_values, std::vector<TF> &v_values, Grid &grid, Bounds &bounds, Func radial_func, bool stop_if_void = true ) {
    struct DataPerThread {
        DataPerThread( std::size_t approx_nb_diracs ) {
            row_items.reserve( 64 );
            offsets  .reserve( approx_nb_diracs );
            columns  .reserve( 5 * approx_nb_diracs );
            values   .reserve( 5 * approx_nb_diracs );
        }

        std::vector<std::pair<TI,TF>> row_items;
        std::vector<TI>               offsets;
        std::vector<TI>               columns;
        std::vector<TF>               values;
    };


    TI nb_diracs = grid.nb_diracs();
    int nb_threads = thread_pool.nb_threads();
    std::vector<DataPerThread> data_per_threads( nb_threads, nb_diracs / nb_threads );
    std::vector<std::pair<int,TI>> pos_in_loc_matrices( nb_diracs ); // num dirac => num_thread, num sub row

    v_values.clear();
    v_values.resize( nb_diracs, 0 );

    int err = grid.for_each_laguerre_cell( [&]( auto &lc, int num_thread ) {
        TI d0_index = *lc.dirac_index;
        TF d0_weight = *lc.dirac_weight;
        Pt d0_center = lc.dirac_center();

        DataPerThread &dpt = data_per_threads[ num_thread ];
        pos_in_loc_matrices[ d0_index ] = { num_thread, dpt.offsets.size() };

        // get local row_items (sorted)
        TF der_0 = 0;
        dpt.row_items.resize( 0 );
        bounds.for_each_intersection( lc, [&]( auto &cp, SpaceFunctions::Constant<TF> space_func ) {
            TF coeff = 0.5 * space_func.coeff;
            v_values[ d0_index ] += space_func.coeff * cp.measure( radial_func.func_for_final_cp_integration(), d0_weight );
            cp.for_each_boundary_measure( radial_func.func_for_final_cp_integration(), [&]( TF boundary_measure, TI d1_index ) {
                if ( d1_index == TI( -1 ) )
                    return;
                if ( d0_index == d1_index ) {
                    der_0 += coeff * boundary_measure / sqrt( d0_weight );
                } else {
                    TI m_num_dirac_1 = d1_index % nb_diracs;
                    Pt d1_center = positions[ m_num_dirac_1 ];
                    if ( std::size_t nu = d1_index / nb_diracs )
                        TODO; // d1_center = transformation( _tranformations[ nu - 1 ], d1_center );

                    TF dist = norm_2( d0_center - d1_center );
                    TF b_der = coeff * boundary_measure / dist;
                    dpt.row_items.emplace_back( m_num_dirac_1, - b_der );
                    der_0 += b_der;
                }
            }, weights[ d0_index ] );

            der_0 += cp.integration_der_wrt_weight( radial_func.func_for_final_cp_integration(), d0_weight );
        } );
        dpt.row_items.emplace_back( d0_index, der_0 );
        std::sort( dpt.row_items.begin(), dpt.row_items.end() );

        // save them in local sub matrix
        dpt.offsets.push_back( dpt.columns.size() );
        for( std::size_t i = 0; i < dpt.row_items.size(); ++i ) {
            if ( i + 1 < dpt.row_items.size() && dpt.row_items[ i ].first == dpt.row_items[ i + 1 ].first ) {
                dpt.row_items[ i + 1 ].second += dpt.row_items[ i ].second;
                continue;
            }
            dpt.columns.push_back( dpt.row_items[ i ].first  );
            dpt.values .push_back( dpt.row_items[ i ].second );
        }
    }, bounds.englobing_convex_polyhedron(), positions, weights, nb_diracs, stop_if_void /*stop if void laguerre cell*/, radial_func.need_ball_cut() );
    if ( err )
        return err;

    if ( stop_if_void )
        for( TF &v : v_values )
            if ( v == 0 )
                return 1;

    // completion of local matrices
    std::size_t nnz = 0;
    for( DataPerThread &dpt : data_per_threads ) {
        dpt.offsets.push_back( dpt.columns.size() );
        nnz += dpt.columns.size();
    }

    // assembly
    m_offsets.resize( nb_diracs + 1 );
    m_columns.reserve( nnz );
    m_values .reserve( nnz );
    m_columns.resize( 0 );
    m_values .resize( 0 );
    for( std::size_t n = 0; n < nb_diracs; ++n ) {
        m_offsets[ n ] = m_columns.size();

        std::size_t lr = pos_in_loc_matrices[ n ].second;
        DataPerThread &dpt = data_per_threads[ pos_in_loc_matrices[ n ].first ];
        m_columns.insert( m_columns.end(), dpt.columns.data() + dpt.offsets[ lr + 0 ], dpt.columns.data() + dpt.offsets[ lr + 1 ] );
        m_values .insert( m_values .end(), dpt.values .data() + dpt.offsets[ lr + 0 ], dpt.values .data() + dpt.offsets[ lr + 1 ] );
    }
    m_offsets.back() = m_columns.size();

    return 0;
}

template<class TI,class TF,class Grid,class Bounds,class Pt>
int get_der_integrals_wrt_weights( std::vector<TI> &m_offsets, std::vector<TI> &m_columns, std::vector<TF> &m_values, std::vector<TF> &v_values, Grid &grid, Bounds &bounds ) {
    return get_der_integrals_wrt_weights( m_offsets, m_columns, m_values, v_values, grid, bounds, FunctionEnum::Unit() );
}

} // namespace sdot
