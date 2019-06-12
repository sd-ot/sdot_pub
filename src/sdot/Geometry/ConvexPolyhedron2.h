#ifndef SDOT_CONVEX_POLYHEDRON_H
#define SDOT_CONVEX_POLYHEDRON_H

//#include "../Integration/SpaceFunctions/Constant.h"
//#include "../Integration/FunctionEnum.h"
//#include "../Display/VtkOutput.h"
#include "Internal/ConvexPolyhedron2NodeBlock.h"
#include "../Support/N.h"

#include <functional>
#include <bitset>

//#define _USE_MATH_DEFINES
//#include <math.h>

namespace sdot {

/**
  Pc must contain
    - dim (2, 3, ...)
    - TI (std::size_t, ...) => index type
    - TF (double, ...) => floating point type
    - CI = Cut id type

  Beware: ball_cuts must be done AFTER the plane_cuts.

  Data layout:
    TF pos_x       [ bs ]
    TF pos_y       [ bs ]
    TF dir_x       [ bs ]
    TF dir_y       [ bs ]
    TF arc_radius  [ bs ] // < 0 if straight line
    TF arc_center_x[ bs ] // arc center x
    TF arc_center_y[ bs ] // arc center y
    TI cut_id      [ bs ]
*/
template<class Pc>
class ConvexPolyhedron2 {
public:
    using                     TF                        = typename Pc::TF; ///< floating point type
    using                     TI                        = typename Pc::TI; ///< index type
    using                     CI                        = typename Pc::CI; ///< cut info
    using                     Pt                        = Point2<TF>;      ///< point type

    static constexpr TI       block_size                = 64;
    using                     Node                      = ConvexPolyhedron2NodeBlock<TF,TI,block_size>;
    struct                    Edge                      { Node *nodes[ 2 ]; }; ///< tmp structure
    // types for the ctor
    struct                    Box                       { Pt p0, p1; };

    /**/                      ConvexPolyhedron2         ( const Box &box, CI cut_id = {} );
    /**/                      ConvexPolyhedron2         ();
    /**/                     ~ConvexPolyhedron2         ();

    // information
    void                      write_to_stream           ( std::ostream &os ) const;
    template<class F> void    for_each_edge             ( const F &f );
    template<class F> void    for_each_node             ( const F &f );
    TI                        nb_nodes                  () const;
    const Node&               node                      ( TI index ) const;
    Node&                     node                      ( TI index );

    //
    void                      resize                    ( TI new_nb_nodes );

    // geometric modifications
    template<int no> bool     plane_cut                 ( Pt origin, Pt dir, CI cut_id, N<no> normal_is_normalized ); ///< return true if effective cut
    bool                      plane_cut                 ( Pt origin, Pt dir, CI cut_id = {} ); ///< return true if effective cut
    void                      ball_cut                  ( Pt center, TF radius, CI cut_id = {} ); ///< beware: only one sphere cut is authorized, and it must be done after all the plane cuts.


private:
    Node*                     nodes;                    ///< aligned data. @see ConvexPolyhedron2
    TI                        size;                     ///< nb nodes
    TI                        rese;                     ///< nb nodes that can be stored without reallocation
};

} // namespace sdot

#include "ConvexPolyhedron2.tcc"

#endif // SDOT_CONVEX_POLYHEDRON_H
