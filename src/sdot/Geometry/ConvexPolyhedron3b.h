#ifndef SDOT_CONVEX_POLYHEDRON_3_H
#define SDOT_CONVEX_POLYHEDRON_3_H

#include "Internal/ConvexPolyhedron3Lt64Face.h"
#include "ConvexPolyhedron.h"
#include <functional>

namespace sdot {

/**
  Pc must contain
    - dim (2, 3, ...)
    - TI (std::size_t, ...) => index type
    - TF (double, ...) => floating point type

  Beware: ball_cuts must be done AFTER the plane_cuts.
*/
template<class Pc>
class alignas( 64 ) ConvexPolyhedron3 : public ConvexPolyhedron {
public:
    using                                Dirac                     = typename Pc::Dirac; ///<
    using                                TF                        = typename Pc::TF;    ///< floating point type
    using                                TI                        = typename Pc::TI;    ///< index type
    using                                CI                        = Dirac *;            ///< cut info
    using                                Pt                        = Point3<TF>;         ///< point type

    static constexpr bool                store_the_normals         = Pc::store_the_normals;
    static constexpr bool                allow_ball_cut            = Pc::allow_ball_cut;
    static constexpr TI                  dim                       = 3;

    using                                Lt64NodeBlock             = ConvexPolyhedron3Lt64NodeBlock<Pc>;
    using                                Lt64FaceBlock             = ConvexPolyhedron3Lt64FaceBlock<Pc>;
    using                                BoundaryItem              = ConvexPolyhedron3Lt64Face<Pc>;
    using                                Face                      = ConvexPolyhedron3Lt64Face<Pc>;
    using                                Node                      = Lt64NodeBlock;

    // types for the ctor
    struct                               Box                       { Box( Pt p0, Pt p1, CI cut_id = nullptr ); ConvexPolyhedron3 cp; };

    /**/                                 ConvexPolyhedron3         ( Pt p0, Pt p1, CI cut_id );
    /**/                                 ConvexPolyhedron3         ();
    /**/                                ~ConvexPolyhedron3         ();

    ConvexPolyhedron3&                   operator=                 ( const ConvexPolyhedron3 &that );
    ConvexPolyhedron3&                   operator=                 ( const Box &that );

    // information
    void                                 write_to_stream           ( std::ostream &os, bool debug = false ) const;
    void                                 display_vtk               ( VtkOutput &vo, const std::vector<TF> &cell_values = {}, Pt offset = TF( 0 ), bool display_both_sides = true ) const;
    TI                                   nb_nodes                  () const;
    TI                                   nb_edges                  () const;
    void                                 check                     () const;

    bool                                 empty                     () const;

    const Node&                          node                      ( TI index ) const;
    Node&                                node                      ( TI index );

    void                                 for_each_boundary_item    ( const std::function<void( const BoundaryItem &boundary_item )> &f ) const;
    void                                 for_each_face             ( const std::function<void( const Face &face )> &f ) const;
    void                                 for_each_node             ( const std::function<void( const Node &node )> &f ) const;

    // geometric modifications
    template<int flags>  void            plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ); ///< return true if effective cut. @see ConvexPolyhedron for the flags
    void                                 plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ); ///< return true if effective cut
    void                                 ball_cut                  ( Pt center, TF radius, CI cut_id = {} ); ///< beware: only one sphere cut is authorized, and it must be done after all the plane cuts.

    //
    TF                                   integral                  () const;

private:
    friend class                         ConvexPolyhedron3Lt64Face<Pc>;
    enum {                               max_nb_nodes = 64 };
    enum {                               max_nb_edges = max_nb_nodes * max_nb_nodes };

    // aligned structures
    ConvexPolyhedron3Lt64NodeBlock<Pc>   nodes;                    ///<
    ConvexPolyhedron3Lt64FaceBlock<Pc>   faces;                    ///<

    int                                  nodes_size;
    int                                  faces_size;

    TF                                   sphere_radius;
    Pt                                   sphere_center;
    CI                                   sphere_cut_id;

    std::uint64_t                        edge_num_cuts [ max_nb_edges ]; ///< to be compared to this->num_cut
    std::uint8_t                         edge_cuts     [ max_nb_edges ]; ///< num node for each possible edge

    std::uint64_t                        num_cut_proc;
};

} // namespace sdot

#include "ConvexPolyhedron3b.tcc"

#endif // SDOT_CONVEX_POLYHEDRON_3_H
