#pragma once

#include "../../Support/Time.h"
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

#ifdef __AVX512F__
template<int nb_bits_per_axis,class Pt>
void make_znodes( std::uint64_t *zcoords, std::uint64_t *indices, std::array<const double *,2> positions, std::size_t nb_diracs, Pt min_point, double grid_length ) {
    RaiiTime rt( "fill_grid_using_zcoords.zcf 512" );
    for( std::uint64_t index = 0; index < nb_diracs; ++index ) {
        zcoords[ index ] = zcoords_for<std::uint64_t,nb_bits_per_axis>( positions, index, min_point, grid_length );
        indices[ index ] = index;
    }
}
#endif


template<int nb_bits_per_axis,class TZ,class TI,class TF,std::size_t dim,class Pt>
void make_znodes( TZ *zcoords, TI *indices, std::array<const TF *,dim> positions, TI nb_diracs, Pt min_point, TF inv_step_length ) {
    RaiiTime rt( "fill_grid_using_zcoords.zcf" );
    for( TI index = 0; index < nb_diracs; ++index ) {
        zcoords[ index ] = zcoords_for<TZ,nb_bits_per_axis>( positions, index, min_point, inv_step_length );
        indices[ index ] = index;
    }
}

//    if ( dim == 2 )
//        //        std::array<TZ,dim> c;
//        //        for( int d = 0; d < dim; ++d )
//        //            c[ d ] = TZ( TF( TZ( 1 ) << nb_bits_per_axis ) * ( pos[ d ] - min_point[ d ] ) / grid_length );

//        //        TZ res = 0;
//        //        switch ( dim ) {
//        //        case 1:
//        //            res = c[ 0 ];
//        //            break;
//        //        case 2:
//        //            for( int o = 0; o < nb_bits_per_axis; o += 8 )
//        //                res |= TZ( morton_256_2D_x[ ( c[ 0 ] >> o ) & 0xFF ] |
//        //                           morton_256_2D_y[ ( c[ 1 ] >> o ) & 0xFF ] ) << dim *  o;
//        //            break;
//        //        case 3:
//        //            for( int o = 0; o < nb_bits_per_axis; o += 8 )
//        //                res |= TZ( morton_256_3D_x[ ( c[ 0 ] >> o ) & 0xFF ] |
//        //                           morton_256_3D_y[ ( c[ 1 ] >> o ) & 0xFF ] |
//        //                           morton_256_3D_z[ ( c[ 2 ] >> o ) & 0xFF ] ) << dim *  o;
//        //            break;
//        //        default:
//        //            TODO;
//        //        }

//        //        return res;
//#endif //  __AVX512F__
//        for( TI index = 0; index < nb_diracs; ++index ) {
//            znodes_keys.push_back( zcoords_for( pt( positions, index ) ) );
//            znodes_inds.push_back( index );
//        }
//#ifdef __AVX512F__
//#endif //  __AVX512F__
//}

}
