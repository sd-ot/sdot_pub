#ifndef SDOT_LGrid_H
#define SDOT_LGrid_H

#include "../Geometry/ConvexPolyhedronTraits.h"
#include "../Support/BumpPointerPool.h"
#include "Internal/CellBoundsTraits.h"
#include <queue>

namespace sdot {

/**
  Makes a grid with cells defined by z-indices.

  We go from neighbors to neighbors until the cell can no longer be cut in the corresponding direction set.

  Weights are handled with a global min_weight variable.

*/
template<class Pc>
class LGrid {
public:
    static constexpr std::size_t   dim                       = Pc::dim;         ///<
    using                          CP                        = typename ConvexPolyhedronTraits<Pc>::type;
    using                          SI                        = typename Pc::SI; ///< signed index type

    using                          TF                        = typename CP::TF; ///< floating point type
    using                          TI                        = typename CP::TI; ///< index type
    using                          CI                        = typename CP::CI; ///< cut info
    using                          Pt                        = typename CP::Pt; ///< point type

    enum {                         homogeneous_weights       = 1 };
    enum {                         ball_cut                  = 2 };

    /* ctor */                     LGrid                     ( std::size_t max_diracs_per_cell = 11 );

    template<int flags> void       update                    ( const Pt *positions, const TF *weights, TI nb_diracs, N<flags>, bool positions_have_changed = true, bool weights_have_changed = true );
    template<int flags> int        for_each_laguerre_cell    ( const std::function<void( CP &lc, TI num, int num_thread )> &f, const CP &starting_lc, N<flags>, bool stop_if_void_lc = false ); ///< version with num_thread

    void                           write_to_stream           ( std::ostream &os ) const;
    void                           display_tikz              ( std::ostream &os, TF scale = 1.0 ) const;
    void                           display                   ( VtkOutput &vtk_output, int disp_weights = 0 ) const; ///< for debug purpose

    // values used by update
    int                            max_diracs_per_cell;
    std::vector<Pt>                translations;

private:
    static constexpr int           nb_bits_per_axis          = 20;
    static constexpr int           sizeof_zcoords            = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    using                          CellBounds                = typename CellBoundsTraits<Pc>::type;
    using                          TZ                        = std::uint64_t; ///< zcoords


    struct                         BaseCell {
        bool                       super_cell                () const { return nb_sub_items < 0; }
        bool                       final_cell                () const { return nb_sub_items > 0; }

        TI                         end_ind_in_fcells;        ///< end index in final cells
        SI                         nb_sub_items;             ///< > 0 => final cell (nb diracs). < 0 => super cell (nb sub cells).
        CellBounds                 bounds;                   ///< pos and weight bounds
    };

    struct                         SuperCell : BaseCell {
        TI                         nb_sub_cells              () const { return - this->nb_sub_items; }

        BaseCell                  *sub_cells[ 1 ];           ///<
    };

    struct                         FinalCell : BaseCell {
        TI                         nb_diracs                 () const { return this->nb_sub_items; }
        TI                         dirac_indice              ( TI index ) const { return dirac_indices[ index ]; }
        TF                         weight                    ( TI index ) const { return weights[ index ]; }
        Pt                         pos                       ( TI index ) const { Pt res; for( std::size_t i = 0; i < dim; ++i ) res[ i ] = positions[ i ][ index ]; return res; }


        TF                        *positions[ dim ];
        TI                        *dirac_indices;
        TF                        *weights;
    };

    struct                         CpAndNum                  { const SuperCell *cell; TI num; };
    struct                         Msi                       { bool operator<( const Msi &that ) const { return dist > that.dist; } Pt center; const BaseCell *cell; TF dist; };

    void                           update_cell_bounds_phase_1( BaseCell *cell, BaseCell **path, int level );
    void                           fill_grid_using_zcoords   ( const Pt *positions, const TF *weights, TI nb_diracs );
    void                           update_the_limits         ( const Pt *positions, TI nb_diracs );
    void                           write_to_stream           ( std::ostream &os, BaseCell *cell, std::string sp ) const;
    template<int flags> bool       can_be_evicted            ( const CP &lc, Pt &c0, TF w0, const CellBoundsP0<Pc> &bounds, N<flags> ) const;
    template<int flags> bool       can_be_evicted            ( const CP &lc, Pt &c0, TF w0, const CellBoundsPpos<Pc> &bounds, N<flags> ) const;
    void                           fill_the_grid             ( const Pt *positions, const TF *weights, TI nb_diracs );
    template<int flags> void       make_lcs_from             ( const std::function<void( CP &, TI num, int num_thread )> &cb, std::priority_queue<Msi> &base_queue, std::priority_queue<Msi> &queue, CP &lc, const FinalCell *cell, const CpAndNum *path, TI path_len, int num_thread, N<flags>, const CP &starting_lc ) const;
    void                           display                   ( VtkOutput &vtk_output, BaseCell *cell, int disp_weights ) const;
    template<int a_n0,int f> void  cut_lc                    ( CP &lc, Pt c0, TF w0, const FinalCell *dell, N<a_n0>, TI n0, N<f> ) const;

    // buffers
    std::vector<TZ>                znodes_keys;              ///< tmp znodes
    std::vector<TI>                znodes_inds;              ///< tmp znodes
    BumpPointerPool                mem_pool;                 ///< store the cells
    std::vector<std::size_t>       rs_tmps;                  ///< for the radix sort

    // grid
    TF                             inv_step_length;
    TI                             nb_final_cells;
    TF                             step_length;
    TF                             grid_length;
    Pt                             min_point;
    Pt                             max_point;
    BaseCell                      *root_cell;
};

} // namespace sdot

#include "LGrid.tcc"

#endif // SDOT_LGrid_H
