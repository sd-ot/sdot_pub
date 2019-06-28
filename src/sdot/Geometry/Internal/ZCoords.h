#pragma once

#include "../../Support/StaticRange.h"
#include "../../Support/ThreadPool.h"
#include "../../Support/Time.h"
#include <immintrin.h>
#include <array>

namespace sdot {

/*
*/
template<class TZ,int dim,int nb_bits_per_axis>
struct ZCoords {
    template                <int num_axis,int _cur_bit = dim * nb_bits_per_axis - 1>
    struct                  _ZcoordsZerosOnAxis {
        static constexpr TZ v_loc = TZ( _cur_bit % dim == num_axis ? 0 : 1 ) << _cur_bit;
        static constexpr TZ value = v_loc | _ZcoordsZerosOnAxis<num_axis,_cur_bit-1>::value;
    };

    template                <int num_axis>
    struct                  _ZcoordsZerosOnAxis<num_axis,-1> {
        static constexpr TZ value = 0;
    };

    /// Ex: axis = 0, dim = 3 (i.e. x) => 000... for level and free_bits ++ 001001001...
    template                <int num_axis,int _cur_bit = dim * nb_bits_per_axis - 1>
    struct                  _ZcoordsOnesOnAxis {
        static constexpr TZ v_loc = TZ( _cur_bit % dim == num_axis ? 1 : 0 ) << _cur_bit;
        static constexpr TZ value = v_loc | _ZcoordsOnesOnAxis<num_axis,_cur_bit-1>::value;
    };
    template                <int num_axis>
    struct                  _ZcoordsOnesOnAxis<num_axis,-1> {
        static constexpr TZ value = 0;
    };
};

extern const std::uint32_t morton_256_2D_x_32[ 256 ];
extern const std::uint32_t morton_256_2D_y_32[ 256 ];
extern const std::uint32_t morton_256_3D_x_32[ 256 ];
extern const std::uint32_t morton_256_3D_y_32[ 256 ];
extern const std::uint32_t morton_256_3D_z_32[ 256 ];

extern const std::uint64_t morton_256_2D_x_64[ 256 ];
extern const std::uint64_t morton_256_2D_y_64[ 256 ];
extern const std::uint64_t morton_256_3D_x_64[ 256 ];
extern const std::uint64_t morton_256_3D_y_64[ 256 ];
extern const std::uint64_t morton_256_3D_z_64[ 256 ];


template<class TZ,int nb_bits_per_axis,class TF,class TI,class Pt>
TZ zcoords_for( std::array<const TF *,1> positions, TI index, Pt min_point, TF inv_step_length ) {
    return TZ( inv_step_length * ( positions[ 0 ][ index ] - min_point.x ) );
}

template<class TZ,int nb_bits_per_axis,class TF,class TI,class Pt>
TZ zcoords_for( std::array<const TF *,2> positions, TI index, Pt min_point, TF inv_step_length ) {
    TZ x = TZ( inv_step_length * ( positions[ 0 ][ index ] - min_point[ 0 ] ) );
    TZ y = TZ( inv_step_length * ( positions[ 1 ][ index ] - min_point[ 1 ] ) );

    TZ res = 0;
    for( int o = 0; o < nb_bits_per_axis; o += 8 )
        res |= TZ( morton_256_2D_x_32[ ( x >> o ) & 0xFF ] |
                   morton_256_2D_y_32[ ( y >> o ) & 0xFF ] ) << 2 * o;
    return res;
}

template<class TZ,int nb_bits_per_axis,class TF,class TI,class Pt>
TZ zcoords_for( std::array<const TF *,3> positions, TI index, Pt min_point, TF inv_step_length ) {
    TZ x = TZ( inv_step_length * ( positions[ 0 ][ index ] - min_point[ 0 ] ) );
    TZ y = TZ( inv_step_length * ( positions[ 1 ][ index ] - min_point[ 1 ] ) );
    TZ z = TZ( inv_step_length * ( positions[ 2 ][ index ] - min_point[ 2 ] ) );

    TZ res = 0;
    for( int o = 0; o < nb_bits_per_axis; o += 8 )
        res |= TZ( morton_256_3D_x_32[ ( x >> o ) & 0xFF ] |
                   morton_256_3D_y_32[ ( y >> o ) & 0xFF ] |
                   morton_256_3D_z_32[ ( z >> o ) & 0xFF ] ) << 3 * o;
    return res;
}

//#ifdef __AVX512F__
//template<int nb_bits_per_axis,class Pt>
//void make_znodes( std::uint64_t *zcoords, std::uint64_t *indices, std::array<const double *,2> positions, std::size_t nb_diracs, Pt min_point, double inv_step_length ) {
//    RaiiTime rt( "fill_grid_using_zcoords.zcf 512" );
//    __m512d isl = _mm512_set1_pd( inv_step_length );
//    __m512d mix = _mm512_set1_pd( min_point.x );
//    __m512d miy = _mm512_set1_pd( min_point.y );
//    __m512i msk = _mm512_set1_epi64( 0xFF );
//    __m512i ind = _mm512_set_epi64( 7, 6, 5, 4, 3, 2, 1, 0 );
//    __m512i c_8 = _mm512_set1_epi64( 8 );

//    std::uint64_t index = 0;
//    for( ; index + 8 <= nb_diracs; index += 8 ) {
//        __m512i x_0 = _mm512_cvtpd_epi64( _mm512_mul_pd( isl, _mm512_sub_pd( _mm512_loadu_pd( positions[ 0 ] + index ), mix ) ) );
//        __m512i y_0 = _mm512_cvtpd_epi64( _mm512_mul_pd( isl, _mm512_sub_pd( _mm512_loadu_pd( positions[ 1 ] + index ), miy ) ) );

//        __m512i res = _mm512_set1_epi64( 0 );
//        StaticRange<(nb_bits_per_axis+7)/8>::for_each( [&]( auto i ) {
//            constexpr int o = i.val * 8;
//            res = _mm512_or_si512( res, _mm512_slli_epi64( _mm512_or_si512(
//                _mm512_i64gather_epi64( _mm512_and_si512( msk, _mm512_srli_epi64( x_0, o ) ), morton_256_2D_x_64, 8 ),
//                _mm512_i64gather_epi64( _mm512_and_si512( msk, _mm512_srli_epi64( y_0, o ) ), morton_256_2D_y_64, 8 )
//            ), 2 * o ) );
//        } );

//        _mm512_storeu_si512( zcoords + index, res );
//        _mm512_storeu_si512( indices + index, ind );

//        ind = _mm512_add_epi64( ind, c_8 );
//    }
//    for( ; index < nb_diracs; ++index ) {
//        zcoords[ index ] = zcoords_for<std::uint64_t,nb_bits_per_axis>( positions, index, min_point, inv_step_length );
//        indices[ index ] = index;
//    }
//}
//#endif

template<int nb_bits_per_axis,class TZ,class TI,class TF,std::size_t dim,class Pt>
void make_znodes( TZ *zcoords, TI *indices, std::array<const TF *,dim> positions, TI nb_diracs, Pt min_point, TF inv_step_length ) {
    RaiiTime rt( "fill_grid_using_zcoords.zcf" );
    TI nb_jobs = thread_pool.nb_threads();
    thread_pool.execute( nb_jobs, [&]( TI num_job, int ) {
        TI beg = ( num_job + 0 ) * nb_diracs / nb_jobs;
        TI end = ( num_job + 1 ) * nb_diracs / nb_jobs;
        for( TI index = beg; index < end; ++index ) {
            zcoords[ index ] = zcoords_for<TZ,nb_bits_per_axis>( positions, index, min_point, inv_step_length );
            indices[ index ] = index;
        }
    } );
}

}
