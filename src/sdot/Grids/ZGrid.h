#ifndef SDOT_ZGRID_H
#define SDOT_ZGRID_H

#include "../Geometry/ConvexPolyhedron2.h"
#include "../Geometry/ConvexPolyhedron3.h"

namespace sdot {

/**
  Makes a grid with cells defined by z-indices.

  We go from neighbors to neighbors until the cell can no longer be cut in the corresponding direction set.

  Weights are handled with a global min_weight variable. If weights are homogenous, homogeneous_weights allows to remove the tests.

*/
template<class Pc>
class ZGrid {
public:
    static constexpr std::size_t    dim                    = Pc::dim;         ///<
    using                           CP2                    = ConvexPolyhedron2<Pc>;
    using                           CP3                    = ConvexPolyhedron3Lt64<Pc>;
    using                           CP                     = typename std::conditional<dim==3,CP3,CP2>::type;

    using                           TF                     = typename CP::TF; ///< floating point type
    using                           TI                     = typename CP::TI; ///< index type
    using                           CI                     = typename CP::Dirac *; ///< cut info
    using                           Pt                     = typename CP::Pt; ///< point type

    enum {                          homogeneous_weights    = 1 };
    enum {                          ball_cut               = 2 };

    /* ctor */                      ZGrid                  ( std::size_t max_diracs_per_cell = 11 );

    template<int flags> void        update                 ( std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs, N<flags>, bool positions_have_changed = true, bool weights_have_changed = true );
    template<int flags,class B> int for_each_laguerre_cell ( const std::function<void( CP &lc, TI num, int num_thread )> &f, const B &starting_lc, std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs, N<flags>, bool stop_if_void_lc = false ); ///< version with num_thread

    void                            display_tikz           ( std::ostream &os, TF scale = 1.0 ) const;
    void                            display                ( VtkOutput &vtk_output ) const; ///< for debug purpose

    // values used by update
    int                             max_diracs_per_cell;
    std::vector<Pt>                 translations;

private:
    static constexpr int            nb_bits_per_axis       = 20;
    static constexpr int            sizeof_zcoords         = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    using                           TZ                     = std::uint64_t; ///< zcoords

    struct                          Cell {
        TI                          dpc_offset;            ///< offsets in grid.dpc_values
        TZ                          zcoords;
        TF                          size;
        Pt                          pos;
    };

    struct                          Grid {
        // std::vector<TI>          dirac_indices;         ///< filled only if blocks.size() > 1
        std::vector<TI>             dpc_values;            ///< index of diracs for each cell
        std::vector<TI>             ng_indices;            ///< list of cell index of direct neighbors
        std::vector<TI>             ng_offsets;            ///< offsets in ng_indices for each cell
        TF                          min_weight;
        TF                          max_weight;
        std::vector<Cell>           cells;
    };

    void                            fill_grid_using_zcoords( std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs );
    void                            repl_zcoords_by_ccoords( const TF *weights );
    void                            update_the_limits      ( std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs );
    void                            update_neighbors       ();
    void                            fill_the_grid          ( std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs );
    template<int flags> TF          min_w_to_cut           ( const CP &lc, const Pt &c0, TF w0, const Cell &cr_cell, N<flags> );
    void                            make_znodes            ( std::array<const TF *,dim> positions, TI nb_diracs );
    template<int d>   TZ            ng_zcoord              ( TZ zcoords, TZ off, N<d> ) const;
    template<int flags> bool        may_cut                ( const CP &lc, const Pt &c0, TF w0, const Cell &cr_cell, N<flags> );
    Pt                              pt                     ( std::array<const TF *,dim> positions, TI index ) const { Pt res; for( std::size_t i = 0; i < dim; ++i ) res[ i ] = positions[ i ][ index ]; return res; }


    // tmp
    std::vector<TZ>                 znodes_keys;           ///< tmp znodes
    std::vector<TI>                 znodes_inds;           ///< tmp znodes
    std::vector<TZ>                 zcells_keys;           ///<
    std::vector<TI>                 zcells_inds;           ///<
    std::vector<std::size_t>        rs_tmps;

    //
    TF                              inv_step_length;
    TF                              step_length;
    TF                              grid_length;
    TF                              min_weight;
    TF                              max_weight;
    Pt                              min_point;
    Pt                              max_point;
    Grid                            grid;                  ///<
};

} // namespace sdot

#include "ZGrid.tcc"

#endif // SDOT_ZGRID_H
