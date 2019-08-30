#pragma once

#include "../Point3.h"
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

    void        foreach_edge  ( const std::function<void(const Edge &edge)> &f ) const {
        if ( Edge e = first_edge ) {
            Node *n0 = e.n0();
            while ( true ) {
                f( e );

                if ( e.n1() == n0 )
                    break;
                e = e.next();
            }
        }
    }

    void        foreach_node  ( const std::function<void(const Node &node)> &f ) const {
        foreach_edge( [&]( const Edge &edge ) { f( *edge.n0() ); } );
    }

    Face       *prev_in_pool; ///<
    Face       *next_in_pool; ///<
    TI          num_cut_proc; ///<
    Edge        first_edge;   ///<
    Pt          normal;       ///<
    CI          cut_id;       ///<
    bool        round;        ///<
};

} // namespace sdot

