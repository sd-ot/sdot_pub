#pragma once

#include <cstdint>
#include <cstring>

namespace sdot {

/**
*/
template<class TI,int bs,int sizeof_TF,bool sizeof_TI_sup_sizeof_TF>
class PaddedType;

//
template<class TI,int bs,int sizeof_TF>
class PaddedType<TI,bs,sizeof_TF,false> {
public:
    TI                   get      () const { return val; }
    void                 set      ( TI n ) { val = n; }

    void                 operator=( TI nval ) { set( nval ); }
    operator             TI       () { return get(); }

    TI                   val;
    char                 _pad[ sizeof_TF * bs - sizeof( TI ) ];
};

//
template<class TI,int bs,int sizeof_TF>
class PaddedType<TI,bs,sizeof_TF,true> {
public:
    static constexpr int n        = ( sizeof( TI ) + sizeof_TF - 1 ) / sizeof_TF;
    struct               Pad      { char values[ sizeof_TF * bs ]; };

    TI                   get      () const;
    void                 set      ( TI n );

    void                 operator=( TI nval ) { set( nval ); }
    operator             TI       () { return get(); }

    Pad                  pads[ n ];
};

template<class TI,int bs,int sizeof_TF>
TI PaddedType<TI,bs,sizeof_TF,true>::get() const {
    TI res = 0;
    int o = 0, n = 0;
    for( ; o + sizeof_TF <= n; o += sizeof_TF, ++n )
        std::memcpy( reinterpret_cast<char *>( &res ) + o, pads[ n ].values, sizeof_TF );
    std::memcpy( reinterpret_cast<char *>( &res ) + o, pads[ n ].values, sizeof( TI ) - o );
    return res;
}

template<class TI,int bs,int sizeof_TF>
void PaddedType<TI,bs,sizeof_TF,true>::set( TI nval ) {
    int o = 0, n = 0;
    for( ; o + sizeof_TF <= n; o += sizeof_TF, ++n )
        std::memcpy( pads[ n ].values, reinterpret_cast<const char *>( &nval ) + o, sizeof_TF );
    std::memcpy( pads[ n ].values, reinterpret_cast<const char *>( &nval ) + o, sizeof( TI ) - o );
}

} // namespace sdot

