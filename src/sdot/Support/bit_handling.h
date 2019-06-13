#pragma once

#include <x86intrin.h>
#include <cstdint>
#include <bitset>
#include <vector>

namespace sdot {

template<std::size_t N>
inline bool none( const std::bitset<N> &bs ) {
    return bs.none();
}

inline bool none( const std::vector<bool> &bs ) {
    for( std::size_t i = 0; i < bs.size(); ++i )
        if ( bs[ i ] )
            return false;
    return true;
}


template<std::size_t N>
inline unsigned popcnt( const std::bitset<N> &bs ) {
    return bs.count();
}

inline unsigned popcnt( const std::vector<bool> &bs ) {
    unsigned res = 0;
    for( std::size_t i = 0; i < bs.size(); ++i )
        res += bs[ i ];
    return res;
}

inline unsigned popcnt( std::uint32_t val ) {
    #ifdef __AVX2__
    return _mm_popcnt_u32( val );
    #else
    unsigned res = 0;
    for( ; val; val /= 2 )
        res += bool( val & 1 );
    return res;
    #endif
}

inline unsigned popcnt( std::uint64_t val ) {
    #ifdef __AVX2__
    return _mm_popcnt_u64( val );
    #else
    unsigned res = 0;
    for( ; val; val /= 2 )
        res += bool( val & 1 );
    return res;
    #endif
}


template<std::size_t N>
inline unsigned tzcnt( const std::bitset<N> &bs ) {
    for( unsigned i = 0; i < bs.size(); ++i )
       if ( bs[ i ] )
           return i;
    return bs.size();
}

inline unsigned tzcnt( const std::vector<bool> &bs ) {
    for( unsigned i = 0; i < bs.size(); ++i )
        if ( bs[ i ] )
           return i;
    return bs.size();
}

inline unsigned tzcnt( std::uint32_t val ) {
    #ifdef __AVX2__
    return _tzcnt_u32( val );
    #else
    if ( val == 0 )
        return 32;
    unsigned res = 0;
    for( ; ( val & 1 ) == 0; ++res )
        val /= 2;
    return res;
    #endif
}

inline unsigned tzcnt( std::uint64_t val ) {
    #ifdef __AVX2__
    return _tzcnt_u64( val );
    #else
    if ( val == 0 )
        return 64;
    unsigned res = 0;
    for( ; ( val & 1 ) == 0; ++res )
        val /= 2;
    return res;
    #endif
}

}
