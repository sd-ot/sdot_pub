#pragma once

#include <ostream>
#include <array>
#include <cmath>

namespace sdot {

/**
*/
template<class TF>
struct Point1 {
    /**/                             Point1         ( const TF *v ) : x( v[ 0 ] ) {}
    /**/                             Point1         ( TF x ) : x( x ) {}
    /**/                             Point1         () {}

    template<class TG>
    operator                         Point1<TG>     () const { return { TG( x ) }; }

    void                             write_to_stream( std::ostream &os ) const { os << x; }
    const TF&                        operator[]     ( std::size_t d ) const { return x; }
    TF&                              operator[]     ( std::size_t d ) { return x; }
    template<class Bq> static Point1 read_from      ( Bq &bq ) { return { TF( bq.read() ) }; }
    template<class Bq> void          write_to       ( Bq &bq ) const { bq << x; }

    Point1&                          operator/=     ( TF v ) { x /= v; return *this ; }

    TF                               x;
};

template<class TF>
inline TF norm_2_p2( Point1<TF> p ) {
    return p.x * p.x;
}


template<class TF>
inline TF norm_2( Point1<TF> p ) {
    using std::sqrt;
    return sqrt( norm_2_p2( p ) );
}

template<class TF>
inline TF dot( Point1<TF> a, Point1<TF> b ) {
    return a.x * b.x;
}

template<class TF>
inline Point1<TF> &operator+=( Point1<TF> &a, Point1<TF> b ) {
    a.x += b.x;
    return a;
}

template<class TF>
inline Point1<TF> &operator-=( Point1<TF> &a, Point1<TF> b ) {
    a.x -= b.x;
    return a;
}

template<class TF>
inline Point1<TF> operator+( Point1<TF> a, Point1<TF> b ) {
    return { a.x + b.x };
}

template<class TF>
inline Point1<TF> operator+( Point1<TF> a, TF b ) {
    return { a.x + b };
}

template<class TF>
inline Point1<TF> operator-( Point1<TF> a, Point1<TF> b ) {
    return { a.x - b.x };
}

template<class TF>
inline Point1<TF> operator-( Point1<TF> a, TF b ) {
    return { a.x - b };
}

template<class TF>
inline Point1<TF> operator-( Point1<TF> a ) {
    return { - a.x };
}

template<class TF>
inline Point1<TF> operator*( TF m, Point1<TF> p ) {
    return { m * p.x };
}

template<class TF>
inline Point1<TF> operator*( Point1<TF> a, Point1<TF> b ) {
    return { a.x * b.x };
}

template<class TF>
inline Point1<TF> operator/( Point1<TF> p, TF d ) {
    return { p.x / d };
}

template<class TF>
inline bool operator==( Point1<TF> p, Point1<TF> q ) {
    return p.x == q.x;
}

template<class TF>
inline bool operator!=( Point1<TF> p, Point1<TF> q ) {
    return p.x != q.x;
}

template<class TF>
inline Point1<TF> min( Point1<TF> p, Point1<TF> q ) {
    using std::min;
    return { min( p.x, q.x ) };
}

template<class TF>
inline Point1<TF> max( Point1<TF> p, Point1<TF> q ) {
    using std::max;
    return { max( p.x, q.x ) };
}

template<class TF>
inline TF max( Point1<TF> p ) {
    using std::max;
    return p.x;
}

template<class TF>
inline Point1<TF> normalized( Point1<TF> p, TF a = 1e-40 ) {
    TF d = norm_2( p ) + a;
    return { p.x / d };
}

template<class TF>
inline Point1<TF> rot90( Point1<TF> p ) {
    return { p.x };
}

template<class TF>
inline Point1<TF> transformation( const std::array<TF,1> &trans, Point1<TF> p ) {
    return { trans[ 0 ] * p.x };
}

template<class TF>
inline TF transformation( const std::array<TF,1> &trans, TF val ) {
    return val;
}

} // namespace sdot
