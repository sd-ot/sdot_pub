#ifndef SDOT_CONVEX_POLYHEDRON_2_Void_H
#define SDOT_CONVEX_POLYHEDRON_2_Void_H

#include "../ConvexPolyhedron.h"
#include <functional>
#include <vector>

namespace sdot {

/**
*/
template<class Pc>
class ConvexPolyhedron<Pc,2,void> {
public:
    using                             TF                        = typename Pc::TF; ///< floating point type
    using                             CI                        = typename Pc::CI; ///< cut info
    using                             Pt                        = Point<TF,2>;     ///< point type
    static constexpr int              dim                       = 2;

    struct                            Node                      { Pt p, n; CI cut_id; TF d; bool outside() const { return d > 0; } };
    struct                            Bound                     { const Node *n0, *n1; CI cut_id() const; template<class TL> void for_each_simplex( const TL &f ) const; };

    /**/                              ConvexPolyhedron          ( Pt pmin, Pt pmax, CI cut_id = {} ); ///< make a box
    /**/                              ConvexPolyhedron          ();

    ConvexPolyhedron&                 operator=                 ( const ConvexPolyhedron &that );

    // information
    void                              write_to_stream           ( std::ostream &os ) const;
    template<class F> void            for_each_bound            ( const F &f ) const;
    template<class F> void            for_each_node             ( const F &f ) const;
    int                               nb_nodes                  () const;
    bool                              empty                     () const;
    Pt                                node                      ( int index ) const;

    // geometric modifications
    template<int flags,class Fu> void plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags>, const Fu &fu ); ///<
    void                              plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ); ///<

    std::vector<Node>                 nodes;                    ///<
    std::vector<Node>                 new_nodes;                ///<
};

} // namespace sdot

#include "ConvexPolyhedron2dVoid.tcc"

#endif // SDOT_CONVEX_POLYHEDRON_2_Void_H
