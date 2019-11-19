#ifndef SDOT_CONVEX_POLYHEDRON_2_H
#define SDOT_CONVEX_POLYHEDRON_2_H

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
class ConvexPolyhedron2 : public ConvexPolyhedron {
public:
    using                                Dirac                     = typename Pc::Dirac; ///<
    using                                TF                        = typename Pc::TF;    ///< floating point type
    using                                TI                        = typename Pc::TI;    ///< index type
    using                                Pt                        = Point2<TF>;         ///< point type
    using                                CI                        = Dirac *;            ///< cut info

    static constexpr bool                store_the_normals         = Pc::store_the_normals; ///< used to test if a point is inside
    static constexpr bool                allow_ball_cut            = Pc::allow_ball_cut;
    static constexpr TI                  block_size                = 64;
    static constexpr TI                  dim                       = 2;

    struct                               BoundaryItem              { std::array<Pt,2> points; TF measure, a0, a1; CI dirac; template<class TL> void add_to_simplex_list( TL &lst ) const; };
    using                                Node                      = ConvexPolyhedron2NodeBlock<TF,TI,CI,block_size,store_the_normals,allow_ball_cut>;
    struct                               Edge                      { Node *nodes[ 2 ]; }; ///< tmp structure

    // types for the ctor
    struct                               Box                       { Pt p0, p1; };

    /**/                                 ConvexPolyhedron2         ( const Box &box, CI cut_id = {} );
    /**/                                 ConvexPolyhedron2         ( ConvexPolyhedron2 &&that );
    /**/                                 ConvexPolyhedron2         ();
    /**/                                ~ConvexPolyhedron2         ();

    ConvexPolyhedron2&                   operator=                 ( const ConvexPolyhedron2 &that );

    // information
    void                                 write_to_stream           ( std::ostream &os ) const;
    template<class F> void               for_each_edge             ( const F &f ) const;
    template<class F> void               for_each_node             ( const F &f ) const;
    void                                 display_vtk               ( VtkOutput &vo, const std::vector<TF> &cell_values = {}, Pt offset = TF( 0 ) ) const;
    TI                                   nb_nodes                  () const;
    bool                                 empty                     () const;
    const Node&                          node                      ( TI index ) const;
    Node&                                node                      ( TI index );

    void                                 for_each_boundary_measure ( const std::function<void( const TF &boundary_measure, Dirac *dirac )> &f, TF weight = 0 ) const;
    void                                 for_each_boundary_item    ( const std::function<void( const BoundaryItem &boundary_item )> &f, TF weight = 0 ) const;

    //
    void                                 resize                    ( TI new_nb_nodes );

    // geometric modifications
    template<int flags>  void            plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ); ///< return true if effective cut
    void                                 plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ); ///< return true if effective cut
    void                                 ball_cut                  ( Pt center, TF radius, CI cut_id = {} ); ///< beware: only one sphere cut is authorized, and it must be done _after all the plane cuts_.

    //
    TF                                   integral                  () const;

    TF                                   sphere_radius;
    Pt                                   sphere_center;
    CI                                   sphere_cut_id;

private:
    template<int f> void                 plane_cut_simd_switch     ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<f>, S<double>, N<8> );
    template<int f,class T,int s> void   plane_cut_simd_switch     ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<f>, S<T>, N<s> );
    template<int f> void                 plane_cut_simd_tzcnt      ( TF cut_dx, TF cut_dy, TF cut_ps, CI cut_id, N<f>, S<double> );
    template<int f,class T> void         plane_cut_simd_tzcnt      ( TF cut_dx, TF cut_dy, TF cut_ps, CI cut_id, N<f>, S<T> );
    template<int f> void                 plane_cut_gen             ( TF cut_dx, TF cut_dy, TF cut_ps, CI cut_id, N<f> );
    template<int f,class B,class D> void plane_cut_gen             ( TF cut_dx, TF cut_dy, TF cut_ps, CI cut_id, N<f>, B &outside, D &distances );

    Node                                *nodes;                    ///< aligned data. @see ConvexPolyhedron2
    TI                                   size;                     ///< nb nodes
    TI                                   rese;                     ///< nb nodes that can be stored without reallocation
};

} // namespace sdot

#include "ConvexPolyhedron2.tcc"

#endif // SDOT_CONVEX_POLYHEDRON_2_H
