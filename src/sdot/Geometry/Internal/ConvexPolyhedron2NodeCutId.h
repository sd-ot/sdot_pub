#pragma once

#include <cstdint>

namespace sdot {

/**
*/
template<class TI,int bs,int sizeof_TF_on_sizeof_TI,int sizeof_TI_on_sizeof_TF>
class ConvexPolyhedron2NodeCutId;

//
template<class TI,int bs>
class ConvexPolyhedron2NodeCutId<TI,bs,1,1> {
public:
    TI   get() const { return val; }
    void set( TI n ) { val = n; }

    TI   val, _pad_val[ bs - 1 ];
};

//
template<int bs>
class ConvexPolyhedron2NodeCutId<std::uint64_t,bs,0,2> {
public:
    using TI = std::uint64_t;
    using TP = std::uint32_t;

    TI    get() const { return TI( v_0 ) | ( TI( v_1 ) << 32 ); }
    void  set( TI n ) { v_0 = TP( n ); v_1 = TP( n >> 32 ); }

    TP    v_0, _pad_v_0[ bs - 1 ];
    TP    v_1, _pad_v_1[ bs - 1 ];
};

//
template<class TI,int bs,int sizeof_TF_on_sizeof_TI>
class ConvexPolyhedron2NodeCutId<TI,bs,sizeof_TF_on_sizeof_TI,0> {
public:
    struct TP { TI v; char p[ sizeof( TI ) * ( sizeof_TF_on_sizeof_TI - 1 ) ]; };

    TI     get() const { return val.v; }
    void   set( TI n ) { val.v = n; }

    TP     val, _pad_val[ bs - 1 ];
};

} // namespace sdot

