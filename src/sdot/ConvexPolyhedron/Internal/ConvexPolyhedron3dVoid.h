#pragma once

#include "../../Support/Point.h"
#include "../../Support/N.h"
#include <functional>
#include <vector>

#include "../ConvexPolyhedron.h"

namespace sdot {

/**
*/
template<class Pc>
class ConvexPolyhedron<Pc,3,void> {
public:
    using                       TF                    = typename Pc::TF;    ///< floating point type
    using                       CI                    = typename Pc::CI;    ///< cut info
    using                       Pt                    = Point<TF,Pc::dim>;  ///< point type

    struct                      BoundaryItem          {};
    struct                      Node                  { Pt pos() const { return p; } Pt p; };
    struct                      Face                  { void for_each_node_index( const std::function<void(int)> &f ) const { for( int n : nodes ) f( n ); } std::vector<int> nodes; CI cut_id; Pt normal; };
    static constexpr int        dim                   = Pc::dim;

    /**/                        ConvexPolyhedron      ( Pt pmin, Pt pmax, CI cut_id = {} ); ///< make a box
    /**/                        ConvexPolyhedron      ();

    void                        write_to_stream       ( std::ostream &os ) const;
    std::size_t                 nb_nodes              () const { return nodes.size(); }
    bool                        empty                 () const { return faces.empty(); }
    Pt                          node                  ( int index ) const { return nodes[ index ].p; }

    void                        for_each_boundary_item( const std::function<void( const BoundaryItem &boundary_item )> &f ) const;
    void                        for_each_face         ( const std::function<void( const Face &face )> &f ) const;
    void                        for_each_node         ( const std::function<void( const Pt &p )> &f ) const;

    template<int flags> void    plane_cut             ( std::array<const TF *,Pc::dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags> ); ///< return the stop cut. @see ConvexPolyhedron for the flags
    void                        plane_cut             ( std::array<const TF *,Pc::dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ); ///< return the stop cut (if < nb_cuts, it means that we have to use another ConvexPolyhedron class)

    std::vector<std::uint64_t>  num_cut_proc_edge;
    std::vector<std::uint64_t>  num_node_edge;
    std::vector<int>            prev_cut_node;
    std::uint64_t               num_cut_proc;
    std::vector<TF>             node_dists;
    std::vector<int>            node_repls;
    std::vector<Node>           new_nodes;
    std::vector<Face>           new_faces;
    std::vector<Node>           nodes;
    std::vector<Face>           faces;
};

} // namespace sdot

#include "ConvexPolyhedron3dVoid.tcc"
