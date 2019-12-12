#pragma once

#include "../Geometry/ConvexPolyhedronTraits.h"
#include <mutex>

namespace sdot {

/**
  Currently, only support constant coeffs per polyhedron
*/
template<class Pc>
class ConvexPolyhedronAssembly {
public:
    static constexpr int   dim                        = Pc::dim;
    using                  TF                         = typename Pc::TF;
    using                  TI                         = typename Pc::TI;

    using                  CP2                        = ConvexPolyhedron2<Pc>;
    using                  CP3                        = ConvexPolyhedron3Lt64<Pc>;
    using                  CP                         = typename std::conditional<dim==3,CP3,CP2>::type;
    using                  Pt                         = typename CP::Pt;

    // modifications
    void                   add_convex_polyhedron      ( const std::vector<Pt> &positions, const std::vector<Pt> &normals, TF coeff = 1.0, TI cut_id = -1 );
    void                   add_box                    ( Pt p0, Pt p1, TF coeff = 1.0, TI cut_id = -1 );

    void                   normalize                  ();

    template               <class Grid,class F>
    void                   for_each_laguerre_cell    ( Grid &grid, const F &func, bool stop_if_void = false ); ///< func( cell, num_thread, space_func )

    // info
    const CP&              englobing_convex_polyhedron() const;
    template<class F> void for_each_intersection      ( CP &cp, const F &f ) const; ///< f( ConvexPolyhedron, SpaceFunction )
    void                   display_boundaries         ( VtkOutput &vtk_output ) const;
    void                   display_coeffs             ( VtkOutput &vtk_output ) const;
    Pt                     min_position               () const;
    Pt                     max_position               () const;
    TF                     measure                    () const;

    TF                     coeff_at                   ( const Pt &pos ) const;

    //
private:
    struct                 Item                       { CP polyhedron; TF coeff; };

    mutable bool           englobing_polyheron_is_up_to_date = false;
    mutable CP             englobing_polyheron;
    mutable std::mutex     mutex;
    std::vector<Item>      items;
};

} // namespace sdot

#include "ConvexPolyhedronAssembly.tcc"
