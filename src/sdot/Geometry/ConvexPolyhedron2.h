#ifndef SDOT_CONVEX_POLYHEDRON_H
#define SDOT_CONVEX_POLYHEDRON_H

//#include "../Integration/SpaceFunctions/Constant.h"
//#include "../Integration/FunctionEnum.h"
#include "Internal/ConvexPolyhedron2NodeBlock.h"
#include "ConvexPolyhedron.h"
#include "../Support/S.h"
#include <functional>

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
    using                                TF                        = typename Pc::TF; ///< floating point type
    using                                TI                        = typename Pc::TI; ///< index type
    using                                CI                        = typename Pc::CI; ///< cut info
    using                                Pt                        = Point2<TF>;      ///< point type

    static constexpr bool                store_the_normals         = Pc::store_the_normals; ///< used to test if a point is inside
    static constexpr bool                allow_ball_cut            = Pc::allow_ball_cut;
    static constexpr TI                  block_size                = 64;
    using                                Node                      = ConvexPolyhedron2NodeBlock<TF,TI,block_size,store_the_normals,allow_ball_cut>;
    struct                               Edge                      { Node *nodes[ 2 ]; }; ///< tmp structure
    struct                               Cut                       { Pt dir; TF dist; CI id; void write_to_stream( std::ostream &os ) const { os << dir << " " << dist; } };

    // types for the ctor
    struct                               Box                       { Pt p0, p1; };

    /**/                                 ConvexPolyhedron2         ( const Box &box, CI cut_id = {} );
    /**/                                 ConvexPolyhedron2         ();
    /**/                                ~ConvexPolyhedron2         ();

    ConvexPolyhedron2&                   operator=                 ( const ConvexPolyhedron2 &that );

    // information
    void                                 write_to_stream           ( std::ostream &os ) const;
    template<class F> void               for_each_edge             ( const F &f ) const;
    template<class F> void               for_each_node             ( const F &f ) const;
    TI                                   nb_nodes                  () const;
    void                                 display                   ( VtkOutput &vo, const std::vector<TF> &cell_values = {}, Pt offset = TF( 0 ) ) const;
    const Node&                          node                      ( TI index ) const;
    Node&                                node                      ( TI index );

    //
    void                                 resize                    ( TI new_nb_nodes );

    // geometric modifications
    template<int flags>                  __attribute__             ((noinline))
    void                                 plane_cut                 ( const Cut *cuts, std::size_t nb_cuts, N<flags> ); ///< return true if effective cut
    void                                 plane_cut                 ( const Cut *cuts, std::size_t nb_cuts ); ///< return true if effective cut
    void                                 ball_cut                  ( Pt center, TF radius, CI cut_id = {} ); ///< beware: only one sphere cut is authorized, and it must be done after all the plane cuts.


private:
    template<int f> void                 plane_cut_simd_switch     ( const Cut *cuts, std::size_t nb_cuts, N<f>, S<double>, S<std::uint64_t> );
    template<int f,class T,class U> void plane_cut_simd_switch     ( const Cut *cuts, std::size_t nb_cuts, N<f>, S<T>, S<U> );
    template<int f> void                 plane_cut_simd_tzcnt      ( const Cut &cut, N<f>, S<double>, S<std::uint64_t> );
    template<int f,class T,class U> void plane_cut_simd_tzcnt      ( const Cut &cut, N<f>, S<T>, S<U> );
    template<int f> void                 plane_cut_gen             ( const Cut &cut, N<f> );
    template<int f,class B,class D> void plane_cut_gen             ( const Cut &cut, N<f>, B &outside, D &distances );

    Node*                                nodes;                    ///< aligned data. @see ConvexPolyhedron2
    TI                                   size;                     ///< nb nodes
    TI                                   rese;                     ///< nb nodes that can be stored without reallocation
};

} // namespace sdot

#include "ConvexPolyhedron2.tcc"

#endif // SDOT_CONVEX_POLYHEDRON_H
