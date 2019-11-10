#ifndef SDOT_LGrid_H
#define SDOT_LGrid_H

#include "Geometry/ConvexPolyhedron2.h"
#include "Support/BumpPointerPool.h"

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
    using                          SI                     = typename Pc::SI; ///< signed index type
    using                          CI                     = typename CP::CI; ///< cut info
    using                          Pt                     = typename CP::Pt; ///< point type

    enum {                         ball_cut               = 2 };

    /* ctor */                     LGrid                  ( std::size_t max_diracs_per_cell = 11 );

    template<int flags> void       update                 ( std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs, N<flags>, bool positions_have_changed = true, bool weights_have_changed = true );
    template<int flags> int        for_each_laguerre_cell ( const std::function<void( CP &lc, TI num, int num_thread )> &f, const CP &starting_lc, std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs, N<flags>, bool stop_if_void_lc = false ); ///< version with num_thread

    void                           write_to_stream        ( std::ostream &os ) const;
    void                           display_tikz           ( std::ostream &os, TF scale = 1.0 ) const;
    void                           display                ( VtkOutput &vtk_output, std::array<const TF *,dim> positions, const TF *weights, int disp_weights = 0 ) const; ///< for debug purpose

    // values used by update
    int                            max_diracs_per_cell;
    std::vector<Pt>                translations;

private:
    static constexpr int           nb_bits_per_axis       = 20;
    static constexpr int           sizeof_zcoords         = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    using                          TZ                     = std::uint64_t; ///< zcoords

    struct                         BaseCell {
        bool                       super_cell             () const { return nb_sub_items < 0; }
        bool                       final_cell             () const { return nb_sub_items > 0; }

        SI                         nb_sub_items;          ///< > 0 => final cell (nb diracs). < 0 => super cell (nb sub cells).
        TF                         max_weight;            ///<
        Pt                         min_pos;               ///< (real) lower left corner
        Pt                         max_pos;               ///< (real) upper right corner
    };

    struct                         SuperCell : BaseCell {
        TI                         nb_sub_cells           () const { return - this->nb_sub_items; }

        BaseCell                  *sub_cells[ 1 ];        ///<
    };

    struct                         FinalCell : BaseCell {
        TI                         nb_diracs              () const { return this->nb_sub_items; }

        TI                         dirac_indices[ 1 ];    ///<
    };

    void                           fill_grid_using_zcoords( std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs );
    void                           update_the_limits      ( std::array<const TF *,dim> positions, TI nb_diracs );
    void                           write_to_stream        ( std::ostream &os, BaseCell *cell, std::string sp ) const;
    void                           fill_the_grid          ( std::array<const TF *,dim> positions, const TF *weights, TI nb_diracs );
    void                           display                ( VtkOutput &vtk_output, std::array<const TF *,dim> positions, const TF *weights, BaseCell *cell, int disp_weights ) const;
    Pt                             pt                     ( std::array<const TF *,dim> positions, TI index ) const { Pt res; for( std::size_t i = 0; i < dim; ++i ) res[ i ] = positions[ i ][ index ]; return res; }

    // buffers
    std::vector<TZ>                znodes_keys;           ///< tmp znodes
    std::vector<TI>                znodes_inds;           ///< tmp znodes
    BumpPointerPool                mem_pool;              ///< store the cells
    std::vector<std::size_t>       rs_tmps;               ///< for the radix sort

    // grid
    TF                             inv_step_length;
    TF                             step_length;
    TF                             grid_length;
    Pt                             min_point;
    Pt                             max_point;
    BaseCell                      *root_cell;
};

} // namespace sdot

#include "LGrid.tcc"

#endif // SDOT_LGrid_H
