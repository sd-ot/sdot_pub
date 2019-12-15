#pragma once

#include "../ConvexPolyhedron.h"
#include <vector>

namespace sdot {

/**
*/
template<class Pc>
class ConvexPolyhedron<Pc,3,void> {
public:
    using                       TF                    = typename Pc::TF;    ///< floating point type
    using                       CI                    = typename Pc::CI;    ///< cut info
    using                       Pt                    = Point<TF,Pc::dim>;  ///< point type
    static constexpr int        dim                   = Pc::dim;

    struct                      Node                  { Pt pos() const { return p; } Pt p; };
    struct                      Face                  { std::vector<int> nodes; CI cut_id; Pt normal; };
    struct                      Bound                 { const ConvexPolyhedron *cp; const Face *face; void for_each_node( const std::function<void(Pt)> &f ) const; };

    /**/                        ConvexPolyhedron      ( Pt pmin, Pt pmax, CI cut_id = {} ); ///< make a box
    /**/                        ConvexPolyhedron      ();

    void                        write_to_stream       ( std::ostream &os ) const;
    std::size_t                 nb_nodes              () const { return nodes.size(); }
    bool                        empty                 () const { return faces.empty(); }
    Pt                          node                  ( int index ) const { return nodes[ index ].p; }

    void                        for_each_bound        ( const std::function<void( const Bound &boundary_item )> &f ) const;
    void                        for_each_node         ( const std::function<void( const Pt &p )> &f ) const;

    template                    <int flags,class Fu>
    void                        plane_cut             ( std::array<const TF *,Pc::dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags>, const Fu &fu ); ///< @see ConvexPolyhedron for the flags
    void                        plane_cut             ( std::array<const TF *,Pc::dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ); ///<

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
