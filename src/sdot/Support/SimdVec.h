#pragma once

#include "SimdVecAggregate.h"
#include "SimdSize.h"
#include "TODO.h"

namespace sdot {

/**
  Generic simd vec.

  (XSIMD would have been a great choice with a better mask handling).
*/
template<class _TF,int _size=SimdSize<_TF>::value> struct SimdVec {
    enum {         size         = _size };
    using          T           = _TF;

    /**/           SimdVec      ( T value ) { for( int i = 0; i < size; ++i ) values[ i ] = value; }
    /**/           SimdVec      () {}

    static void    store_aligned( T *data, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) data[ i ] = vec.values[ i ]; }
    template       <class GF>
    static SimdVec load_aligned ( const GF *data ) { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = data[ i ]; return res; }

    static SimdVec iota         () { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = i; return res; }

    SimdVec        operator+    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] + that.values[ i ]; return res; }
    SimdVec        operator-    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] - that.values[ i ]; return res; }
    SimdVec        operator*    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] * that.values[ i ]; return res; }
    SimdVec        operator/    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] / that.values[ i ]; return res; }

    std::uint64_t  operator>    ( const SimdVec &that ) const { std::uint64_t res = 0; for( int i = 0; i < size; ++i ) res |= std::uint64_t( values[ i ] > that.values[ i ] ) << i; return res;  }

    SimdVec        operator<<   ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] << that.values[ i ]; return res; }

    SimdVec        operator&    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] & that.values[ i ]; return res; }

    std::uint64_t  nz           () const { std::uint64_t res = 0; for( int i = 0; i < size; ++i ) res |= std::uint64_t( values[ i ] != 0 ) << i; return res; }

    const T       *begin        () const { return values; }
    const T       *end          () const { return values + size; }

    T              values       [ size ];
};

#define SIMD_AGGREGATE( TF, TARGET_SIZE, USED_SIZE ) \
    template<> struct SimdVec<TF,TARGET_SIZE> : SimdVecAggregate<SimdVec<TF,TARGET_SIZE>,SimdVec<TF,USED_SIZE>,TARGET_SIZE/USED_SIZE> { using SimdVecAggregate<SimdVec<TF,TARGET_SIZE>,SimdVec<TF,USED_SIZE>,TARGET_SIZE/USED_SIZE>::SimdVecAggregate; }

#ifdef __AVX512F__
    template<> struct SimdVec<double,8> {
        enum {         size         = 8 };
        using          T           = double;

        /**/           SimdVec      ( __m512d values ) : values( values ) {}
        /**/           SimdVec      ( double value ) { values = _mm512_set1_pd( value ); }
        /**/           SimdVec      () {}

        static void    store_aligned( double *data, const SimdVec &vec ) { _mm512_store_pd( data, vec.values ); }
        static SimdVec load_aligned ( const double *data ) { return _mm512_load_pd( data ); }

        static SimdVec iota         () { SimdVec res; for( int i = 0; i < 8; ++i ) res.values[ i ] = i; return res; }

        SimdVec        operator+    ( const SimdVec &that ) const { return values + that.values; }
        SimdVec        operator-    ( const SimdVec &that ) const { return values - that.values; }
        SimdVec        operator*    ( const SimdVec &that ) const { return values * that.values; }
        SimdVec        operator/    ( const SimdVec &that ) const { return values / that.values; }

        std::uint64_t  operator>    ( const SimdVec &that ) const { return _mm512_cmp_pd_mask( values, that.values, _CMP_GT_OQ );  }

        const T       *begin        () const { return reinterpret_cast<const T *>( this ); }
        const T       *end          () const { return begin() + size; }

        __m512d        values;
    };

    //            __m128i blo = _mm_load_si128( reinterpret_cast<const __m128i *>( fnodes.data() ) ); // load the 16 indices (in 8 bit)
    //            __m512i bou = _mm512_set1_epi64( ou );
    //            __m512i bex = _mm512_cvtepi8_epi64( blo ); // 8 bits to 64 bits indices (first part)
    //            __m512i bsh = _mm512_sllv_epi64( _mm512_set1_epi64( 1 ), bex ); // load the first 8 indices to 64 bits
    //            __m512i ban = _mm512_and_epi64( bsh, bou );
    //            std::uint16_t ouf = ( _mm512_cmpneq_epi64_mask( ban, _mm512_setzero_si512() ) << 3 ) + ( nb_fnodes - 3 );

    SIMD_AGGREGATE( double, 16, 8 );
#endif

#ifdef __AVX2__
    template<> struct SimdVec<double,4> {
        enum {         size         = 4 };
        using          T           = double;

        /**/           SimdVec      ( __m256d values ) : values( values ) {}
        /**/           SimdVec      ( T value ) { values = _mm256_set1_pd( value ); }
        /**/           SimdVec      () {}

        static SimdVec iota         () { SimdVec res; for( int i = 0; i < 4; ++i ) res.values[ i ] = i; return res; }

        static void    store_aligned( T *data, const SimdVec &vec ) { _mm256_store_pd( data, vec.values ); }
        static SimdVec load_aligned ( const T *data ) { return _mm256_load_pd( data ); }

        SimdVec        operator+    ( const SimdVec &that ) const { return values + that.values; }
        SimdVec        operator-    ( const SimdVec &that ) const { return values - that.values; }
        SimdVec        operator*    ( const SimdVec &that ) const { return values * that.values; }
        SimdVec        operator/    ( const SimdVec &that ) const { return values / that.values; }

        std::uint64_t  operator>    ( const SimdVec &that ) const { __m256d c = _mm256_cmp_pd( values, that.values, _CMP_GT_OQ ); return _mm256_movemask_pd( c );  }

        const T       *begin        () const { return reinterpret_cast<const T *>( this ); }
        const T       *end          () const { return begin() + size; }

        __m256d        values;
    };

    SIMD_AGGREGATE( double,  8, 4 );
    SIMD_AGGREGATE( double, 16, 4 );

    // ------------------------------------------------------------------------------------------------------------------
    template<> struct SimdVec<std::uint64_t,4> {
        enum {         size         = 4 };
        using          T            = std::uint64_t;

        /**/           SimdVec      ( __m256i values ) : values( values ) {}
        /**/           SimdVec      ( T value ) { values = _mm256_set1_epi64x( value ); }
        /**/           SimdVec      () {}

        static SimdVec iota         () { SimdVec res; for( int i = 0; i < 4; ++i ) res.values[ i ] = i; return res; }

        static void    store_aligned( T *data, const SimdVec &vec ) { _mm256_store_epi64( data, vec.values ); }
        static SimdVec load_aligned ( const T *data ) { return _mm256_load_si256( reinterpret_cast<const __m256i *>( data ) ); }

        SimdVec        operator+    ( const SimdVec &that ) const { return values + that.values; }
        SimdVec        operator-    ( const SimdVec &that ) const { return values - that.values; }
        SimdVec        operator*    ( const SimdVec &that ) const { return values * that.values; }
        SimdVec        operator/    ( const SimdVec &that ) const { return values / that.values; }

        // std::uint64_t  operator> ( const SimdVec &that ) const { __m256d c = _mm256_cmp_epipd( values, that.values, _CMP_GT_OQ ); return _mm256_movemask_pd( c );  }

        SimdVec        operator<<   ( const SimdVec &that ) const { return _mm256_sllv_epi64( values, that.values ); }

        const T       *begin        () const { return reinterpret_cast<const T *>( this ); }
        const T       *end          () const { return begin() + size; }

        __m256i        values;
    };

    SIMD_AGGREGATE( std::uint64_t,  8, 4 );
    SIMD_AGGREGATE( std::uint64_t, 16, 4 );
#endif


} // namespace sdot
