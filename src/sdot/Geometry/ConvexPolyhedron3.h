#ifndef SDOT_CONVEX_POLYHEDRON_H
#define SDOT_CONVEX_POLYHEDRON_H

#include "Internal/ConvexPolyhedron3NodeBlock.h"
#include "Internal/ConvexPolyhedron3EdgeBlock.h"
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
class ConvexPolyhedron3 : public ConvexPolyhedron {
public:
    using                                TF                        = typename Pc::TF; ///< floating point type
    using                                TI                        = typename Pc::TI; ///< index type
    using                                CI                        = typename Pc::CI; ///< cut info
    using                                Pt                        = Point3<TF>;      ///< point type

    static constexpr bool                store_the_normals         = Pc::store_the_normals; ///< used to test if a point is inside
    static constexpr bool                allow_ball_cut            = Pc::allow_ball_cut;
    static constexpr TI                  block_size                = 64;
    static constexpr TI                  dim                       = 3;
    struct                               BoundaryItem              { std::array<Pt,2> points; TF measure, a0, a1; CI id; template<class TL> void add_simplex_list( TL &lst ) const; };
    using                                Node                      = ConvexPolyhedron3NodeBlock<TF,TI,CI,block_size,store_the_normals,allow_ball_cut>;
    using                                Edge                      = ConvexPolyhedron3EdgeBlock<TF,TI,block_size,allow_ball_cut>;
    struct                               Face                      { TI num_in_edge_beg; TI num_in_edge_len; Pt normal; CI cut_id; bool round; };

    // types for the ctor
    struct                               Box                       { Pt p0, p1; };

    /**/                                 ConvexPolyhedron3         ( const Box &box, CI cut_id = {} );
    /**/                                 ConvexPolyhedron3         ();
    /**/                                ~ConvexPolyhedron3         ();

    ConvexPolyhedron3&                   operator=                 ( const ConvexPolyhedron3 &that );

    // information
    void                                 write_to_stream           ( std::ostream &os ) const;
    template<class F> void               for_each_edge             ( const F &f ) const;
    template<class F> void               for_each_node             ( const F &f ) const;
    TI                                   nb_nodes                  () const;
    TI                                   nb_edges                  () const;
    void                                 display                   ( VtkOutput &vo, const std::vector<TF> &cell_values = {}, Pt offset = TF( 0 ) ) const;
    bool                                 empty                     () const;

    const Node&                          node                      ( TI index ) const;
    Node&                                node                      ( TI index );

    const Edge&                          edge                      ( TI index ) const;
    Edge&                                edge                      ( TI index );

    void                                 for_each_boundary_item    ( const std::function<void( const BoundaryItem &boundary_item )> &f, TF weight = 0 ) const;

    //
    void                                 set_nb_nodes              ( TI new_nb_nodes );
    void                                 set_nb_edges              ( TI new_nb_edges );

    // geometric modifications
    template<int flags>  void            plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ); ///< return true if effective cut
    void                                 plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ); ///< return true if effective cut
    void                                 ball_cut                  ( Pt center, TF radius, CI cut_id = {} ); ///< beware: only one sphere cut is authorized, and it must be done after all the plane cuts.

    //
    TF                                   integral                  () const;

    TF                                   sphere_radius;
    Pt                                   sphere_center;
    CI                                   sphere_cut_id;

private:
    std::vector<int>                     num_in_edges;             ///< 2 * real_num + 1 * reverse
    TI                                   nodes_size;               ///< nb nodes
    TI                                   nodes_rese;               ///< nb nodes that can be stored without reallocation
    TI                                   edges_size;               ///< nb edges
    TI                                   edges_rese;               ///< nb edges that can be stored without reallocation
    Node*                                nodes;                    ///< aligned data
    Edge*                                edges;                    ///< aligned data
    std::vector<Face>                    faces;                    ///<
};

} // namespace sdot

#include "ConvexPolyhedron3.tcc"

#endif // SDOT_CONVEX_POLYHEDRON_H
