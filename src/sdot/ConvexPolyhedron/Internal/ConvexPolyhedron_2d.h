#ifndef SDOT_CONVEX_POLYHEDRON_2D_H
#define SDOT_CONVEX_POLYHEDRON_2D_H

#include "../ConvexPolyhedron.h"
#include "../../Support/Point.h"
#include "../../Support/N.h"
#include <functional>

namespace sdot {

/**
*/
template<class Pc>
class ConvexPolyhedron<Pc,2,void> {
public:
    using                      TF                        = typename Pc::TF; ///< floating point type
    using                      CI                        = typename Pc::CI; ///< cut info
    using                      Pt                        = Point<TF,2>;     ///< point type

    static constexpr int       dim                       = 2;

    // struct                  Bound                     { const NodeBlock *nodes; int n0, n1; CI cut_id() const; template<class TL> void for_each_simplex( const TL &f ) const; };

    /**/                       ConvexPolyhedron          ( Pt pmin, Pt pmax, CI cut_id = {} ); ///< make a box
    /**/                       ConvexPolyhedron          ();
    /**/                      ~ConvexPolyhedron          ();

    ConvexPolyhedron&          operator=                 ( const ConvexPolyhedron &that );

    // information
    void                       write_to_stream           ( std::ostream &os ) const;
    template<class F> void     for_each_bound            ( const F &f ) const;
    template<class F> void     for_each_node             ( const F &f ) const;
    int                        nb_nodes                  () const;
    bool                       empty                     () const;
    Pt                         node                      ( int index ) const;

    // geometric modifications
    template<int flags> void   plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_ids, int nb_cuts, N<flags> ); ///<
    void                       plane_cut                 ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_ids, int nb_cuts ); ///<

private:
    typedef                    TF                        ATF __attribute__ ((aligned(64))); // __declspec(align(64)) float A[1000];
    typedef                    CI                        ACI __attribute__ ((aligned(64)));


    void                       reserve_without_cp        ( int nb_nodes );

    int                        nodes_size;               ///< nb nodes
    int                        nodes_rese;               ///< room for nodes
    ATF*                       pxs;                      ///< node positions, x axis (aligned in memory)
    ATF*                       pys;                      ///< node positions, y axis (aligned in memory)
    ATF*                       nxs;                      ///< normals, x axis (aligned in memory)
    ATF*                       nys;                      ///< normals, y axis (aligned in memory)
    ACI*                       cis;                      ///< cut ids (aligned in memory)
    ATF*                       ds;                       ///< distances from the current cutting plane (aligned in memory)
};

} // namespace sdot

#include "ConvexPolyhedron_2d.tcc"

#endif // SDOT_CONVEX_POLYHEDRON_2D_H
