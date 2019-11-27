#ifndef SDOT_LGrid_H
#define SDOT_LGrid_H

#include "../Geometry/ConvexPolyhedronTraits.h"
#include "Internal/LGridOutOfCoreCell.h"
#include "Internal/CellBoundsTraits.h"
#include "Internal/LGridFinalCell.h"
#include "Internal/LGridSuperCell.h"
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
    // synonyms
    using                          Dirac                       = typename Pc::Dirac; ///< info for each dirac
    static constexpr std::size_t   dim                         = Pc::dim;            ///<
    using                          TF                          = typename Pc::TF;    ///< floating point type
    using                          TI                          = typename Pc::TI;    ///< index type
    using                          Pt                          = typename Pc::Pt;    ///< point type

    // constructed or deduced types
    struct                         TraversalFlags              { bool stop_if_void_lc = false, mod_weights = false; };
    using                          OutOfCoreCell               = LGridOutOfCoreCell<Pc>;
    struct                         DisplayFlags                { TF weight_elevation = 0; bool display_cells = false, display_boxes = true; };
    using                          FinalCell                   = LGridFinalCell<Pc>;
    using                          SuperCell                   = LGridSuperCell<Pc>;
    using                          BaseCell                    = LGridBaseCell<Pc>;
    using                          CP                          = typename ConvexPolyhedronTraits<Pc>::type;
    using                          Cb                          = std::function<void( const Dirac *diracs, TI nb_diracs, bool ptrs_survive_the_call )>; ///<

    // contruction methods
    /* ctor */                     LGrid                       ( std::size_t max_diracs_per_cell = 11 );
    /* dtor */                    ~LGrid                       ();

    void                           construct                   ( const std::function<void( const Cb &cb )> &f ); ///< generic case: diracs come by chunk (that fit in memory or not)
    void                           construct                   ( const Dirac *diracs, TI nb_diracs ); ///< simple case: one gives all the diracs at once

    void                           update_after_mod_weights     (); ///< update grid info after modification of diracs->weights (not necessary if done in a traversal where mod_weight was specified in the flags)

    // traversal/information
    template<class SLC> int        for_each_laguerre_cell      ( const std::function<void( CP &lc, Dirac &dirac, int num_thread )> &f, const SLC &starting_lc, TraversalFlags traversal_flags = {} ) const; ///< version with num_thread
    void                           for_each_final_cell         ( const std::function<void( FinalCell &cell, int num_thread )> &f, TraversalFlags traversal_flags = {} ) const;
    void                           for_each_dirac              ( const std::function<void( Dirac &d, int num_thread)> &f, TraversalFlags traversal_flags = {} ) const;
    TI                             nb_diracs                   () const { return nb_diracs_tot; }

    // display
    void                           write_to_stream             ( std::ostream &os ) const;
    void                           display_tikz                ( std::ostream &os, DisplayFlags display_flags = {} ) const;
    void                           display_vtk                 ( VtkOutput &vtk_output, DisplayFlags display_flags = {} ) const; ///< for debug purpose

    // values used by update
    TI                             max_diracs_per_cell;
    std::size_t                    max_ram_per_sst;
    std::vector<Pt>                translations;

private:
    static constexpr int           nb_bits_per_axis            = 20;
    static constexpr int           sizeof_zcoords              = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    using                          CellBounds                  = typename CellBoundsTraits<Pc>::type;
    using                          LocalSolver                 = typename CellBounds::LocalSolver;
    using                          TZ                          = std::uint64_t; ///< zcoords

    enum {                         homogeneous_weights         = 1 };
    enum {                         ball_cut                    = 2 };

    struct                         TmpLevelInfo                { void clr() { num_sub_cell = 0; nb_sub_cells = 0; ls.clr(); } BaseCell *sub_cells[ 1 << dim ]; TI  num_sub_cell, nb_sub_cells; LocalSolver ls; };
    struct                         CpAndNum                    { SuperCell *cell; TI num; };
    struct                         Msi                         { bool operator<( const Msi &that ) const { return dist > that.dist; } Pt center; BaseCell *cell; TF dist; };

    void                           get_grid_dims_and_dirac_ptrs( const std::function<void(const Cb &cb)> &f );
    void                           for_each_final_cell_mono_thr( const std::function<void( FinalCell &cell, CpAndNum *path, TI path_len )> &f, TI beg_num_cell, TI end_num_cell ) const;
    void                           update_after_mod_weights_rec( BaseCell *cell, LocalSolver *local_solvers, int level );
    void                           update_cell_bounds_phase_1  ( BaseCell *cell, BaseCell **path, int level );
    void                           fill_grid_using_zcoords     ( const Dirac *diracs, TI nb_diracs );
    void                           compute_sst_limits          ( const std::function<void(const Cb &cb)> &f );
    void                           make_zind_limits            ( std::vector<TI> &zind_indices, std::vector<TZ> &zind_limits, const std::function<void(const LGrid::Cb &)> &f );
    void                           write_to_stream             ( std::ostream &os, BaseCell *cell, std::string sp ) const;
    template<int flags> bool       can_be_evicted              ( const CP &lc, Pt &c0, TF w0, const CellBoundsP0<Pc> &bounds, N<flags> ) const;
    template<int flags> bool       can_be_evicted              ( const CP &lc, Pt &c0, TF w0, const CellBoundsPpos<Pc> &bounds, N<flags> ) const;
    void                           make_the_cells              ( const std::function<void(const Cb &cb)> &f );
    template<int f,class SLC> void make_lcs_from               ( const std::function<void( CP &, Dirac &dirac, int num_thread )> &cb, std::priority_queue<Msi> &base_queue, std::priority_queue<Msi> &queue, CP &lc, FinalCell *cell, const CpAndNum *path, TI path_len, int num_thread, N<f>, const SLC &starting_lc ) const;
    void                           display_vtk                 ( VtkOutput &vtk_output, BaseCell *cell, DisplayFlags display_flags ) const;
    void                           push_cell                   ( TI l, TZ &prev_z, TI level, TmpLevelInfo *level_info, TI &index, const Dirac **zn_ptrs, TZ *zn_keys );

    template<int a_n0,int f> void  cut_lc                      ( CP &lc, Point2<TF> c0, TF w0, FinalCell *dell, N<a_n0>, TI n0, N<f> ) const;
    template<int a_n0,int f> void  cut_lc                      ( CP &lc, Point3<TF> c0, TF w0, FinalCell *dell, N<a_n0>, TI n0, N<f> ) const;

    // buffers
    std::vector<TZ>                znodes_keys;                ///< buffer for some zcoords
    std::vector<const Dirac *>     znodes_ptrs;                ///< buffer for some dirac pointers
    std::vector<Dirac>             tmp_diracs;                 ///< buffer to store the dirac for a sub-part of the grid
    std::vector<std::size_t>       rs_tmps;                    ///< for the radix sort

    //
    bool                           use_diracs_from_cb;         ///< true if possible to use a simple (stable) dirac pointer to construct the grid
    const Dirac                   *diracs_from_cb;             ///< the simple (stable) dirac pointer to construct the grid (if possible)
    TI                             nb_diracs_tot;

    // grid
    std::deque<OutOfCoreCell>      out_of_core_cells;          ///<
    TF                             inv_step_length;
    TI                             nb_final_cells;
    BumpPointerPool                mem_pool_cells;             ///< to store the cells
    TF                             step_length;
    TF                             grid_length;
    Pt                             min_point;
    Pt                             max_point;
    BaseCell                      *root_cell;
};

} // namespace sdot

#include "LGrid.tcc"

#endif // SDOT_LGrid_H
