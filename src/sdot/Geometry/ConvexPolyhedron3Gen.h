#ifndef CONVEXPOLYHEDRON3GEN_H
#define CONVEXPOLYHEDRON3GEN_H

#include "ConvexPolyhedron3Lt64.h"

namespace sdot {

/**
  Generic 3D convex polyhedron.

  Not optimized. The most common case are covered by ConvexPolyhedron3Lt64, which is far more optimized than ConvexPolyhedron3Gen
*/
template<class Pc>
class ConvexPolyhedron3Gen : public ConvexPolyhedron {
public:
    using                      Dirac                 = typename Pc::Dirac; ///<
    using                      TF                    = typename Pc::TF;    ///< floating point type
    using                      TI                    = typename Pc::TI;    ///< index type
    using                      CI                    = Dirac *;            ///< cut info
    using                      Pt                    = Point3<TF>;         ///< point type

    static constexpr int       dim                   = 3;
    struct                     BoundaryItem          {};
    struct                     Node                  { Pt pos() const { return p; } Pt p; };
    struct                     Face                  { std::vector<int> nodes; };

    /**/                       ConvexPolyhedron3Gen  ();

    ConvexPolyhedron3Gen&      operator=             ( const ConvexPolyhedron3Lt64<Pc> &cp );

    void                       display_vtk           ( VtkOutput &vo, const std::vector<TF> &cell_values = {}, Pt offset = TF( 0 ), bool display_both_sides = true ) const;
    std::size_t                nb_nodes              () const { return nodes.size(); }
    bool                       empty                 () const { return faces.empty(); }
    const Node&                node                  ( TI index ) const { return nodes[ index ]; }
    Node&                      node                  ( TI index ) { return nodes[ index ]; }

    void                       for_each_boundary_item( const std::function<void( const BoundaryItem &boundary_item )> &f ) const;
    void                       for_each_face         ( const std::function<void( const Face &face )> &f ) const;
    void                       for_each_node         ( const std::function<void( const Node &node )> &f ) const;

    template<int flags> void   plane_cut             ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ); ///< return the stop cut. @see ConvexPolyhedron for the flags
    void                       plane_cut             ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ); ///< return the stop cut (if < nb_cuts, it means that we have to use another ConvexPolyhedron class)

    std::vector<std::uint64_t> num_cut_proc_edge;
    std::vector<std::uint64_t> num_node_edge;
    std::vector<int>           prev_cut_node;
    std::uint64_t              num_cut_proc;
    std::vector<TF>            node_dists;
    std::vector<int>           node_repls;
    std::vector<Node>          new_nodes;
    std::vector<Face>          new_faces;
    std::vector<Node>          nodes;
    std::vector<Face>          faces;
};

} // namespace sdot

#include "ConvexPolyhedron3Gen.tcc"

#endif // CONVEXPOLYHEDRON3GEN_H
