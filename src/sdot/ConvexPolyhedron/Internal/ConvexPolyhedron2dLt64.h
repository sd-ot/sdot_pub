#ifndef SDOT_CONVEX_POLYHEDRON_2_LT64_H
#define SDOT_CONVEX_POLYHEDRON_2_LT64_H

#include "../ConvexPolyhedron.h"
#include <functional>

namespace sdot {

/**
*/
template<class Pc>
class ConvexPolyhedron<Pc,2,ConvexPolyhedronOpt::Lt64> {
public:
    using                             TF                        = typename Pc::TF; ///< floating point type
    using                             CI                        = typename Pc::CI; ///< cut info
    using                             Pt                        = Point<TF,2>;     ///< point type

    static constexpr int              dim                       = 2;
    static constexpr int              bs                        = 64;

    using                             NodeBlock                 = ConvexPolyhedron2dLt64_NodeBlock<TF,CI,bs,true>;
    struct                            Bound                     { const NodeBlock *nodes; int n0, n1; CI cut_id() const; template<class TL> void for_each_simplex( const TL &f ) const; };

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
    template<int flags,class Fu> void plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_ids, std::size_t nb_cuts, N<flags>, const Fu &fu ); ///<
    void                              plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_ids, std::size_t nb_cuts ); ///<

private:
    NodeBlock                         nodes;                    ///< aligned data
    int                               nodes_size;               ///< nb nodes

    ConvexPolyhedron<Pc,2>            cp_gen;                   ///<
};

} // namespace sdot

#include "ConvexPolyhedron2dLt64.tcc"

#endif // SDOT_CONVEX_POLYHEDRON_2_LT64_H
