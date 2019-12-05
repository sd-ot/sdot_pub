#ifndef SDOT_LGrid_H
#define SDOT_LGrid_H

#include "../Geometry/ConvexPolyhedronTraits.h"
#include "Internal/LGridCell.h"
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
    struct                         DisplayFlags                { TF weight_elevation = 0; bool display_cells = false, display_boxes = true; };
    using                          Cell                        = LGridCell<Pc>;
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
    void                           for_each_final_cell         ( const std::function<void( Cell &cell, int num_thread )> &f, TraversalFlags traversal_flags = {} ) const;
    void                           for_each_dirac              ( const std::function<void( Dirac &d, int num_thread)> &f, TraversalFlags traversal_flags = {} ) const;
    TI                             nb_diracs                   () const { return nb_diracs_tot; }

    // display
    void                           write_to_stream             ( std::ostream &os ) const;
    void                           display_tikz                ( std::ostream &os, DisplayFlags display_flags = {} ) const;
    void                           display_vtk                 ( VtkOutput &vtk_output, DisplayFlags display_flags = {} ) const; ///< for debug purpose

    // values used by update
    TI                             nb_final_cells_per_ooc_file;
    TI                             max_diracs_per_cell;
    std::size_t                    max_ram_per_sst;
    std::size_t                    max_usable_ram;
    std::vector<Pt>                translations;
    std::string                    ooc_dir;

private:
    static constexpr int           nb_bits_per_axis            = 20;
    static constexpr int           sizeof_zcoords              = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    using                          CellBounds                  = typename CellBoundsTraits<Pc>::type;
    using                          LocalSolver                 = typename CellBounds::LocalSolver;
    using                          TZ                          = std::uint64_t; ///< zcoords

    enum {                         homogeneous_weights         = 1 };
    enum {                         ball_cut                    = 2 };

    struct                         OutOfCoreInfo               { std::string filename; bool in_memory = true, saved = false; };
    struct                         TmpLevelInfo                { void clr() { num_scell = 0; nb_scells = 0; ls.clr(); } Cell *scells[ 1 << dim ]; TI num_scell, nb_scells; LocalSolver ls; };
    struct                         CpAndNum                    { Cell *cell; TI num; };
    struct                         Msi                         { bool operator<( const Msi &that ) const { return dist > that.dist; } Pt center; Cell *cell; TF dist; };

    void                           get_grid_dims_and_dirac_ptrs( const std::function<void(const Cb &cb)> &f );
    void                           for_each_final_cell_mono_thr( const std::function<void( Cell &cell, CpAndNum *path, TI path_len )> &f, TI beg_num_cell, TI end_num_cell, Cell *root_cell = 0 ) const;
    void                           update_after_mod_weights_rec( Cell *cell, LocalSolver *local_solvers, int level );
    void                           update_cell_bounds_phase_1  ( Cell *cell, Cell **path, int level );
    void                           fill_grid_using_zcoords     ( const Dirac *diracs, TI nb_diracs );
    void                           compute_sst_limits          ( const std::function<void(const Cb &cb)> &f );
    void                           free_an_out_of_core_cell    ();
    void                           make_zind_limits            ( std::vector<TI> &zind_indices, std::vector<TZ> &zind_limits, const std::function<void(const LGrid::Cb &)> &f );
    void                           write_to_stream             ( std::ostream &os, Cell *cell, std::string sp ) const;
    //    LGridBaseCell<Pc>       *deserialize_rec             ( char *base, std::size_t off ) const;
    template<int flags> bool       can_be_evicted              ( const CP &lc, Pt &c0, TF w0, const CellBoundsP0<Pc> &bounds, N<flags> ) const;
    template<int flags> bool       can_be_evicted              ( const CP &lc, Pt &c0, TF w0, const CellBoundsPpos<Pc> &bounds, N<flags> ) const;
    void                           make_the_cells              ( const std::function<void(const Cb &cb)> &f );
    template<int f,class SLC> void make_lcs_from               ( const std::function<void( CP &, Dirac &dirac, int num_thread )> &cb, std::priority_queue<Msi> &base_queue, std::priority_queue<Msi> &queue, CP &lc, Cell *cell, const CpAndNum *path, int path_len, int num_thread, N<f>, const SLC &starting_lc ) const;
    //    std::size_t              serialize_rec               ( std::ostream &os, std::size_t &len, BaseCell *cell );
    void                           display_vtk                 ( VtkOutput &vtk_output, Cell *cell, DisplayFlags display_flags ) const;
    // void                        deserialize                 ( OutOfCoreCell *cell ) const;
    void                           push_cell                   ( TI l, TZ &prev_z, TI level, TmpLevelInfo *level_info, TI &index, const Dirac **zn_ptrs, TZ *zn_keys );
    void                           free_ooc                    ( TI num_ooc );
    // void                        serialize                   ( OutOfCoreCell *cell );
    void                           reset                       ();

    template<int a_n0,int f> void  cut_lc                      ( CP &lc, Point2<TF> c0, TF w0, Cell *dell, N<a_n0>, TI n0, N<f> ) const;
    template<int a_n0,int f> void  cut_lc                      ( CP &lc, Point3<TF> c0, TF w0, Cell *dell, N<a_n0>, TI n0, N<f> ) const;

    // buffers
    std::vector<TZ>                znodes_keys;                ///< buffer for some zcoords
    std::vector<const Dirac *>     znodes_ptrs;                ///< buffer for some dirac pointers
    std::vector<Dirac>             tmp_diracs;                 ///< buffer to store the dirac for a sub-part of the grid
    std::vector<std::size_t>       rs_tmps;                    ///< for the radix sort

    //
    bool                           use_diracs_from_cb;         ///< true if possible to use a simple (stable) dirac pointer to construct the grid
    const Dirac                   *diracs_from_cb;             ///< the simple (stable) dirac pointer to construct the grid (if possible)
    TI                             nb_diracs_tot;

    // cache
    std::vector<OutOfCoreInfo>     out_of_core_infos;          ///<
    TI                             used_fcell_ram;             ///<
    TI                             used_scell_ram;             ///<
    std::size_t                    nb_filenames;               ///<
    BumpPointerPool                pool_scells;                ///<
    BumpPointerPool                pool_fcells;                ///<

    // grid
    TF                             inv_step_length;
    TI                             nb_final_cells;
    TF                             step_length;
    TF                             grid_length;
    Pt                             min_point;
    Pt                             max_point;
    Cell                          *root_cell;
};

} // namespace sdot

#include "LGrid.tcc"

#endif // SDOT_LGrid_H
