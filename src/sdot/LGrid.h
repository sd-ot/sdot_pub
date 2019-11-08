#ifndef SDOT_LGrid_H
#define SDOT_LGrid_H

#include "Geometry/ConvexPolyhedron2.h"

namespace sdot {

/**
  Makes a grid with cells defined by z-indices.

  We go from neighbors to neighbors until the cell can no longer be cut in the corresponding direction set.

  Weights are handled with a global min_weight variable.

*/
template<class Pc>
class LGrid {
public:
    static constexpr std::size_t   dim                    = Pc::dim;         ///<
    using                          CP2                    = ConvexPolyhedron2<Pc>;
    using                          CP3                    = ConvexPolyhedron2<Pc>;
    using                          CP                     = typename std::conditional<dim==3,CP3,CP2>::type;

    using                          TF                     = typename CP::TF; ///< floating point type
    using                          TI                     = typename CP::TI; ///< index type
    using                          CI                     = typename CP::CI; ///< cut info
    using                          Pt                     = typename CP::Pt; ///< point type

    enum {                         ball_cut               = 2 };

    /* ctor */                     LGrid                  ( std::size_t max_diracs_per_cell = 11 );

    template<int flags> void       update                 ( std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs, N<flags>, bool positions_have_changed = true, bool weights_have_changed = true );
    template<int flags> int        for_each_laguerre_cell ( const std::function<void( CP &lc, TI num, int num_thread )> &f, const CP &starting_lc, std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs, N<flags>, bool stop_if_void_lc = false ); ///< version with num_thread

    void                           write_to_stream        ( std::ostream &os ) const;
    void                           display_tikz           ( std::ostream &os, TF scale = 1.0 ) const;
    void                           display                ( VtkOutput &vtk_output, int disp_weights = 0 ) const; ///< for debug purpose

    // values used by update
    int                            max_diracs_per_cell;
    std::vector<Pt>                translations;

private:
    static constexpr int           nb_bits_per_axis       = 20;
    static constexpr int           sizeof_zcoords         = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    using                          TZ                     = std::uint64_t; ///< zcoords

    struct                         Cell {
        TI                         dpc_offset;            ///< offsets in dpc_indices
        TI                         msi_offset;            ///< offsets in msi_info
        TF                         max_weight;
        TZ                         zcoords;               ///<
        TF                         size;                  ///<
        Pt                         pos;                   ///< lower left corner
    };

    struct                         MsiInfo {              ///<
        TI                         cell_indices[ 3 ];     ///< cell indices of the first degree sub-cells
        TI                         num_in_parent;         ///< in {0,1,2,3}: num sub-cell in parent cell
        TI                         parent_index;          ///< index in msi_infos. If no parent, parent_index is equal to the msi_info index
        TF                         max_weight;            ///<
    };

    void                           fill_grid_using_zcoords( std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs );
    void                           make_the_cell_list     ( const TF *weights );
    void                           update_the_limits      ( std::array<const TF *,dim> positions, TI nb_diracs );
    void                           fill_the_grid          ( std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs );
    template<int flags> TF         min_w_to_cut           ( const CP &lc, const Pt &c0, TF w0, const Cell &cr_cell, N<flags> );
    template<int d>   TZ           ng_zcoord              ( TZ zcoords, TZ off, N<d> ) const;
    template<int flags> bool       may_cut                ( const CP &lc, const Pt &c0, TF w0, const Cell &cr_cell, N<flags> );
    Pt                             pt                     ( std::array<const TF *,dim> positions, TI index ) const { Pt res; for( std::size_t i = 0; i < dim; ++i ) res[ i ] = positions[ i ][ index ]; return res; }

    // buffers
    std::vector<TZ>                znodes_keys;           ///< tmp znodes
    std::vector<TI>                znodes_inds;           ///< tmp znodes
    std::vector<std::size_t>       rs_tmps;               ///< for the radix sort

    // grid content
    std::vector<TI>                dpc_indices;           ///< dirac indices for each cell
    std::vector<MsiInfo>           msi_infos;             ///< multi-scale info
    std::vector<Cell>              cells;

    // grid dimensions
    TF                             inv_step_length;
    TF                             step_length;
    TF                             grid_length;
    Pt                             min_point;
    Pt                             max_point;
};

} // namespace sdot

#include "LGrid.tcc"

#endif // SDOT_LGrid_H
