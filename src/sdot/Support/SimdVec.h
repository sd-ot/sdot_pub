#pragma once

#include "SimdSize.h"
#include <cstdint>
#include "TODO.h"

namespace sdot {

/**
  Generic simd vec.

  (XSIMD would have been a great choice with a better mask handling).
*/
template<class TF,int size=SimdSize<TF>::value> struct SimdVec {
    /**/           SimdVec      ( TF value ) { for( int i = 0; i < size; ++i ) values[ i ] = value; }
    /**/           SimdVec      () {}

    static void    store_aligned( TF *data, const SimdVec &vec ) { for( int i = 0; i < size; ++i ) data[ i ] = vec.values[ i ]; }
    template       <class GF>
    static SimdVec load_aligned ( const GF *data ) { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = data[ i ]; return res; }

    SimdVec        operator+    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] + that.values[ i ]; return res; }
    SimdVec        operator-    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] - that.values[ i ]; return res; }
    SimdVec        operator*    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] * that.values[ i ]; return res; }
    SimdVec        operator/    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] / that.values[ i ]; return res; }

    std::uint64_t  operator>    ( const SimdVec &that ) const { std::uint64_t res = 0; for( int i = 0; i < size; ++i ) res |= std::uint64_t( values[ i ] > that.values[ i ] ) << i; return res;  }

    SimdVec        operator<<   ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] << that.values[ i ]; return res; }

    SimdVec        operator&    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] & that.values[ i ]; return res; }

    std::uint64_t  nz           () const { std::uint64_t res = 0; for( int i = 0; i < size; ++i ) res |= std::uint64_t( values[ i ] != 0 ) << i; return res; }

    const TF      *begin        () const { return values; }
    const TF      *end          () const { return values + size; }

    TF             values       [ size ];
};

#ifdef __AVX512F__
    template<> struct SimdVec<double,8> {
        /**/           SimdVec      ( __m512d values ) : values( values ) {}
        /**/           SimdVec      ( double value ) { values = _mm512_set1_pd( value ); }
        /**/           SimdVec      () {}

        static void    store_aligned( double *data, const SimdVec &vec ) { _mm512_store_pd( data, vec.values ); }
        static SimdVec load_aligned ( const double *data ) { return _mm512_load_pd( data ); }

        SimdVec        operator+    ( const SimdVec &that ) const { return values + that.values; }
        SimdVec        operator-    ( const SimdVec &that ) const { return values - that.values; }
        SimdVec        operator*    ( const SimdVec &that ) const { return values * that.values; }
        SimdVec        operator/    ( const SimdVec &that ) const { return values / that.values; }

        std::uint64_t  operator>    ( const SimdVec &that ) const { return _mm512_cmp_pd_mask( values, that.values, _CMP_GT_OQ );  }

        __m512d        values;
    };

    //            __m128i blo = _mm_load_si128( reinterpret_cast<const __m128i *>( fnodes.data() ) ); // load the 16 indices (in 8 bit)
    //            __m512i bou = _mm512_set1_epi64( ou );
    //            __m512i bex = _mm512_cvtepi8_epi64( blo ); // 8 bits to 64 bits indices (first part)
    //            __m512i bsh = _mm512_sllv_epi64( _mm512_set1_epi64( 1 ), bex ); // load the first 8 indices to 64 bits
    //            __m512i ban = _mm512_and_epi64( bsh, bou );
    //            std::uint16_t ouf = ( _mm512_cmpneq_epi64_mask( ban, _mm512_setzero_si512() ) << 3 ) + ( nb_fnodes - 3 );

#endif

#ifdef __AVX2__
    template<> struct SimdVec<double,4> {
        /**/           SimdVec      ( __m256d values ) : values( values ) {}
        /**/           SimdVec      ( double value ) { values = _mm256_set1_pd( value ); }
        /**/           SimdVec      () {}

        static void    store_aligned( double *data, const SimdVec &vec ) { _mm256_store_pd( data, vec.values ); }
        static SimdVec load_aligned ( const double *data ) { return _mm256_load_pd( data ); }

        SimdVec        operator+    ( const SimdVec &that ) const { return values + that.values; }
        SimdVec        operator-    ( const SimdVec &that ) const { return values - that.values; }
        SimdVec        operator*    ( const SimdVec &that ) const { return values * that.values; }
        SimdVec        operator/    ( const SimdVec &that ) const { return values / that.values; }

        std::uint64_t  operator>    ( const SimdVec &that ) const { __m256d c = _mm256_cmp_pd( values, that.values, _CMP_GT_OQ ); return _mm256_movemask_pd( c );  }

        __m256d        values;
    };
#endif


} // namespace sdot
