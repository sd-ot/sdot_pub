#pragma once

#include "../../Support/Point3.h"
#include "../../Support/TODO.h"
#include <functional>
#include "Simplex.h"

namespace sdot {

template<class Carac> class ConvexPolyhedron3NodeBlock;
template<class Carac> class ConvexPolyhedron3Edge;

/**
  Data layout:
*/
template<class Carac>
class ConvexPolyhedron3Face {
public:
    using       Node           = ConvexPolyhedron3NodeBlock<Carac>;
    using       Edge           = ConvexPolyhedron3Edge<Carac>;
    using       Face           = ConvexPolyhedron3Face<Carac>;
    using       CI             = typename Carac::Dirac *;
    using       TF             = typename Carac::TF;
    using       TI             = typename Carac::TI;
    using       Pt             = Point3<TF>;

    void        foreach_simplex( const std::function<void( const Simplex<TF,3,2> &simplex )> &f ) const;
    void        foreach_edge   ( const std::function<void( const Edge &edge )> &f ) const;
    void        foreach_node   ( const std::function<void( const Node &node )> &f ) const;
    const Node *first_node     () const;
    TF          flat_area      () const;
    TF          mass           () const;

    Face       *prev_in_pool;  ///<
    Face       *next_in_pool;  ///<
    TI          num_cut_proc;  ///<
    Face       *prev_marked;   ///<
    Edge        first_edge;    ///<
    Pt          normal;        ///<
    CI          cut_id;        ///<
    bool        round;         ///<
};


// -------------------------------------------------------------------------------------------------------------------------
template<class Carac>
void ConvexPolyhedron3Face<Carac>::foreach_simplex( const std::function<void( const Simplex<TF,3,2> &simplex )> &f ) const {
    if ( Edge e = first_edge ) {
        Node *n0 = e.n0();
        while ( true ) {
            e = e.next();
            if ( e.n0() == n0 )
                break;
            f( { { n0->pos(), e.n0()->pos(), e.n1()->pos() } } );
        }
    }
}

template<class Carac>
void ConvexPolyhedron3Face<Carac>::foreach_edge( const std::function<void (const ConvexPolyhedron3Face::Edge &)> &f ) const {
    if ( Edge e = first_edge ) {
        for( Node *n0 = e.n0(); ; ) {
            Edge n = e.next();
            f( e );

            if ( n.n0() == n0 )
                break;
            e = n;
        }
    }
}

template<class Carac>
void ConvexPolyhedron3Face<Carac>::foreach_node( const std::function<void (const ConvexPolyhedron3Face::Node &)> &f ) const {
    foreach_edge( [&]( const Edge &edge ) { f( *edge.n0() ); } );
}

template<class Carac>
const typename ConvexPolyhedron3Face<Carac>::Node *ConvexPolyhedron3Face<Carac>::first_node() const {
    return first_edge.n0();
}

template<class Carac>
typename ConvexPolyhedron3Face<Carac>::TF ConvexPolyhedron3Face<Carac>::flat_area() const {
    TF res = 0;
    if ( Edge e = first_edge ) {
        Node *n0 = e.n0();
        while ( true ) {
            e = e.next();
            if ( e.n0() == n0 )
                break;
            res += norm_2( cross_prod( e.n0()->pos() - n0->pos(), e.n1()->pos() - n0->pos() ) );
        }
    }
    return res / 2;
}

template<class Carac>
typename ConvexPolyhedron3Face<Carac>::TF ConvexPolyhedron3Face<Carac>::mass() const {
    if ( Carac::allow_ball_cut )
        TODO;
    return flat_area();
}

} // namespace sdot

