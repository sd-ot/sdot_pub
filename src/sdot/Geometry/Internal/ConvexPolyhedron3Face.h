#pragma once

#include "../../Support/Point3.h"
#include <functional>

namespace sdot {

template<class Carac> class ConvexPolyhedron3NodeBlock;
template<class Carac> class ConvexPolyhedron3Edge;

/**
  Data layout:
*/
template<class Carac>
class ConvexPolyhedron3Face {
public:
    using       Node          = ConvexPolyhedron3NodeBlock<Carac>;
    using       Edge          = ConvexPolyhedron3Edge<Carac>;
    using       Face          = ConvexPolyhedron3Face<Carac>;
    using       TF            = typename Carac::TF;
    using       TI            = typename Carac::TI;
    using       CI            = typename Carac::CI;
    using       Pt            = Point3<TF>;

    void        foreach_edge  ( const std::function<void(const Edge &edge)> &f ) const;
    void        foreach_node  ( const std::function<void(const Node &node)> &f ) const;

    Face       *prev_in_pool; ///<
    Face       *next_in_pool; ///<
    TI          num_cut_proc; ///<
    Face       *prev_marked;  ///<
    Edge        first_edge;   ///<
    Pt          normal;       ///<
    CI          cut_id;       ///<
    bool        round;        ///<
};


// -------------------------------------------------------------------------------------------------------------------------
template<class Carac>
void ConvexPolyhedron3Face<Carac>::foreach_edge(const std::function<void (const ConvexPolyhedron3Face::Edge &)> &f) const {
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
void ConvexPolyhedron3Face<Carac>::foreach_node(const std::function<void (const ConvexPolyhedron3Face::Node &)> &f) const {
    foreach_edge( [&]( const Edge &edge ) { f( *edge.n0() ); } );
}

} // namespace sdot

