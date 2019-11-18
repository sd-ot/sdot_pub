#ifndef SDOT_CONVEX_POLYHEDRON_3_H
#define SDOT_CONVEX_POLYHEDRON_3_H

#include "Internal/ConvexPolyhedron3NodeBlock.h"
#include "Internal/ConvexPolyhedron3Face.h"
#include "ConvexPolyhedron.h"

#include "../Support/IntrusivePool.h"
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
*/
template<class Pc>
class ConvexPolyhedron3 : public ConvexPolyhedron {
public:
    using                                Dirac                     = typename Pc::Dirac; ///<
    using                                TF                        = typename Pc::TF;    ///< floating point type
    using                                TI                        = typename Pc::TI;    ///< index type
    using                                CI                        = Dirac *;            ///< cut info
    using                                Pt                        = Point3<TF>;         ///< point type

    static constexpr bool                store_the_normals         = Pc::store_the_normals;
    static constexpr bool                allow_ball_cut            = Pc::allow_ball_cut;
    static constexpr TI                  block_size                = ConvexPolyhedron3NodeBlock<Pc>::bs;
    static constexpr TI                  dim                       = 3;

    struct                               BoundaryItem              { std::array<Pt,2> points; TF measure, a0, a1; CI id; template<class TL> void add_simplex_list( TL &lst ) const; };
    using                                Node                      = ConvexPolyhedron3NodeBlock<Pc>;
    using                                Edge                      = ConvexPolyhedron3Edge<Pc>;
    using                                Face                      = ConvexPolyhedron3Face<Pc>;

    // types for the ctor
    struct                               Box                       { Pt p0, p1; CI cut_id = {}; };

    /**/                                 ConvexPolyhedron3         ( const Box &box );
    /**/                                 ConvexPolyhedron3         ();
    /**/                                ~ConvexPolyhedron3         ();

    ConvexPolyhedron3&                   operator=                 ( const ConvexPolyhedron3 &that );
    ConvexPolyhedron3&                   operator=                 ( const Box &box );

    // information
    void                                 write_to_stream           ( std::ostream &os, bool debug = false ) const;
    template<class F> void               for_each_face             ( const F &f ) const;
    template<class F> void               for_each_edge             ( const F &f ) const;
    template<class F> void               for_each_node             ( const F &f ) const;
    void                                 display_vtk               ( VtkOutput &vo, const std::vector<TF> &cell_values = {}, Pt offset = TF( 0 ), bool display_both_sides = true ) const;
    TI                                   nb_nodes                  () const;
    TI                                   nb_edges                  () const;
    void                                 check                     () const;

    bool                                 empty                     () const;

    const Node&                          node                      ( TI index ) const;
    Node&                                node                      ( TI index );

    void                                 for_each_boundary_item    ( const std::function<void( const BoundaryItem &boundary_item )> &f, TF weight = 0 ) const;

    //
    void                                 rese_nb_nodes             ( TI new_nb_nodes );
    void                                 set_nb_nodes              ( TI new_nb_nodes );
    Node                                *new_node                  ( Pt pos );

    // geometric modifications
    template<int flags>  void            plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ); ///< return true if effective cut. @see ConvexPolyhedron for the flags
    void                                 plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ); ///< return true if effective cut
    void                                 ball_cut                  ( Pt center, TF radius, CI cut_id = {} ); ///< beware: only one sphere cut is authorized, and it must be done after all the plane cuts.

    //
    TF                                   integral                  () const;

    TF                                   sphere_radius;
    Pt                                   sphere_center;
    CI                                   sphere_cut_id;

private:
    template<int flags>  void            plane_cut_lt_64           ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI */*cut_id*/, std::size_t nb_cuts, N<flags> );
    template<int flags>  void            plane_cut_mt_64           ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI */*cut_id*/, std::size_t nb_cuts, N<flags> );
    void                                 set_box                   ( const Box &box );

    TI                                   num_cut_proc;             ///<
    TI                                   nodes_size;               ///< nb nodes
    TI                                   nodes_rese;               ///< nb nodes that can be stored without reallocation
    Node*                                nodes;                    ///< aligned data
    IntrusivePool<Face,2048>             faces;                    ///<
};

} // namespace sdot

#include "ConvexPolyhedron3.tcc"

#endif // SDOT_CONVEX_POLYHEDRON_3_H
