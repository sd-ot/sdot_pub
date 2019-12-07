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
    static SimdVec load_aligned ( const TF *data ) { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = data[ i ]; return res; }

    SimdVec        operator+    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] + that.values[ i ]; return res; }
    SimdVec        operator-    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] - that.values[ i ]; return res; }
    SimdVec        operator*    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] * that.values[ i ]; return res; }
    SimdVec        operator/    ( const SimdVec &that ) const { SimdVec res; for( int i = 0; i < size; ++i ) res.values[ i ] = values[ i ] / that.values[ i ]; return res; }

    std::uint64_t  operator>    ( const SimdVec &that ) const { std::uint64_t res = 0; for( int i = 0; i < size; ++i ) res |= std::uint64_t( values[ i ] > that.values[ i ] ) << i; return res;  }

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
