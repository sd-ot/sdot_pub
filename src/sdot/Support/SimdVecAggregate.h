#pragma once

#include <cstdint>

namespace sdot {

/**
  TS => Target Simd Vec.
  US => Used Simd Vec
*/
template<class TS,class US,int ns> struct SimdVecAggregate {
    using                T               = typename US::T;

    /**/                 SimdVecAggregate( T value ) { for( int i = 0; i < ns; ++i ) vecs[ i ] = value; }
    /**/                 SimdVecAggregate() {}

    static void          store_aligned   ( T *data, const TS &vec ) { for( int i = 0; i < ns; ++i ) US::store_aligned( data + i * US::size, vec.vecs[ i ] ); }
    template             <class GF>
    static TS            load_aligned    ( const GF *data ) { TS res; for( int i = 0; i < ns; ++i ) res.vecs[ i ] = US::load_aligned( data + i * US::size ); return res; }

    static TS            iota            () { TS res; US l = US::iota(); for( int i = 0; i < ns; ++i ) { res.vecs[ i ] = l; l = l + US::size; } return res; }

    TS                   operator+       ( const TS &that ) const { TS res; for( int i = 0; i < ns; ++i ) res.vecs[ i ] = vecs[ i ] + that.vecs[ i ]; return res; }
    TS                   operator-       ( const TS &that ) const { TS res; for( int i = 0; i < ns; ++i ) res.vecs[ i ] = vecs[ i ] - that.vecs[ i ]; return res; }
    TS                   operator*       ( const TS &that ) const { TS res; for( int i = 0; i < ns; ++i ) res.vecs[ i ] = vecs[ i ] * that.vecs[ i ]; return res; }
    TS                   operator/       ( const TS &that ) const { TS res; for( int i = 0; i < ns; ++i ) res.vecs[ i ] = vecs[ i ] / that.vecs[ i ]; return res; }

    TS                   operator<<      ( const TS &that ) const { TS res; for( int i = 0; i < ns; ++i ) res.vecs[ i ] = vecs[ i ] << that.vecs[ i ]; return res; }
    TS                   operator&       ( const TS &that ) const { TS res; for( int i = 0; i < ns; ++i ) res.vecs[ i ] = vecs[ i ] & that.vecs[ i ]; return res; }

    std::uint64_t        operator>       ( const TS &that ) const { std::uint64_t res = 0; for( int i = 0, a = 0; i < ns; ++i, a += US::size ) res |= ( vecs[ i ] > that.vecs[ i ] ) << a; return res;  }
    std::uint64_t        neg             () const { std::uint64_t res = 0; for( int i = 0, a = 0; i < ns; ++i, a += US::size ) res |= vecs[ i ].neg() << a; return res; }
    std::uint64_t        nz              () const { std::uint64_t res = 0; for( int i = 0, a = 0; i < ns; ++i, a += US::size ) res |= vecs[ i ].nz() << a; return res; }

    const T             *begin           () const { return vecs[ 0 ].begin(); }
    const T             *end             () const { return vecs[ ns - 1 ].end(); }

    US                   vecs            [ ns ];
};

} // namespace sdot
