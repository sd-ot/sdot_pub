#ifndef SDOT_LGrid_H
#define SDOT_LGrid_H

#include "../Geometry/ConvexPolyhedronTraits.h"
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
    struct                         DisplayFlags                { TF weight_elevation = 0; };
    using                          FinalCell                   = LGridFinalCell<Pc>;
    using                          SuperCell                   = LGridSuperCell<Pc>;
    using                          BaseCell                    = LGridBaseCell<Pc>;
    using                          CP                          = typename ConvexPolyhedronTraits<Pc>::type;
    using                          Cb                          = std::function<void( const Dirac *diracs, TI nb_diracs, bool ptrs_survive_the_call )>; ///<

    // contruction methods
    /* ctor */                     LGrid                       ( std::size_t max_diracs_per_cell = 11 );

    void                           construct                   ( const std::function<void( const Cb &cb )> &f ); ///< generic case: diracs come by chunk (that fit in memory or not)
    void                           construct                   ( const Dirac *diracs, TI nb_diracs ); ///< simple case: one gives all the diracs at once

    void                           update_grid_wrt_weights     (); ///< update grid info after modification of diracs->weights (not necessary if mod_weight is specified in traversal_flags)

    // traversal/information
    int                            for_each_laguerre_cell      ( const std::function<void( CP &lc, Dirac &dirac, int num_thread )> &f, const CP &starting_lc, TraversalFlags traversal_flags = {} ); ///< version with num_thread
    void                           for_each_final_cell         ( const std::function<void( FinalCell &cell, int num_thread )> &f, TraversalFlags traversal_flags = {} );
    void                           for_each_dirac              ( const std::function<void( Dirac &d, int num_thread)> &f, TraversalFlags traversal_flags = {} );
    TI                             nb_diracs                   () const { return nb_diracs_tot; }

    // display
    void                           write_to_stream             ( std::ostream &os ) const;
    void                           display_tikz                ( std::ostream &os, DisplayFlags display_flags = {} ) const;
    void                           display_vtk                 ( VtkOutput &vtk_output, DisplayFlags display_flags = {} ) const; ///< for debug purpose

    // values used by update
    TI                             max_diracs_per_cell;
    TI                             max_diracs_per_sst;
    std::vector<Pt>                translations;

private:
    static constexpr int           nb_bits_per_axis            = 20;
    static constexpr int           sizeof_zcoords              = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    using                          CellBounds                  = typename CellBoundsTraits<Pc>::type;
    using                          LocalSolver                 = typename CellBounds::LocalSolver;
    struct                         DiracPn                     { const Dirac *diracs; TI nb_diracs; };
    using                          TZ                          = std::uint64_t; ///< zcoords

    enum {                         homogeneous_weights         = 1 };
    enum {                         ball_cut                    = 2 };

    struct                         SstLimits                   { TZ beg_zcoords, end_zcoords; TI nb_diracs; int level; };
    struct                         CpAndNum                    { SuperCell *cell; TI num; };
    struct                         Msi                         { bool operator<( const Msi &that ) const { return dist > that.dist; } Pt center; BaseCell *cell; TF dist; };

    void                           get_grid_dims_and_dirac_ptrs( const std::function<void(const Cb &cb)> &f );
    void                           for_each_final_cell_mono_thr( const std::function<void( FinalCell &cell, CpAndNum *path, TI path_len )> &f, TI beg_num_cell, TI end_num_cell );
    void                           update_grid_wrt_weights_rec ( BaseCell *cell, LocalSolver *local_solvers, int level );
    void                           make_znodes_with_1ppwn_ssst ( const SstLimits &sst, const Dirac *diracs, TI nb_diracs ); ///< several sst case
    void                           make_znodes_with_1ppwn_1sst ( const Dirac *diracs, TI nb_diracs );                          ///< only one sst case (=> no need to make a test)
    void                           update_cell_bounds_phase_1  ( BaseCell *cell, BaseCell **path, int level );
    void                           fill_grid_using_zcoords     ( const Dirac *diracs, TI nb_diracs );
    void                           compute_sst_limits          ( const std::function<void(const Cb &cb)> &f );
    template<class Ps> void        make_the_cells_for          ( const SstLimits &sst, Ps ps );
    void                           write_to_stream             ( std::ostream &os, BaseCell *cell, std::string sp ) const;
    TI                            *znodes_seconds              ( const DiracPn &              ) { return znodes_inds.data(); }
    std::pair<TI,TI>              *znodes_seconds              ( const std::vector<DiracPn> & ) { return znodes_pnds.data(); }
    Dirac                         *znodes_seconds              ( int                          ) { return znodes_vals.data(); }
    template<int flags> bool       can_be_evicted              ( const CP &lc, Pt &c0, TF w0, const CellBoundsP0<Pc> &bounds, N<flags> ) const;
    template<int flags> bool       can_be_evicted              ( const CP &lc, Pt &c0, TF w0, const CellBoundsPpos<Pc> &bounds, N<flags> ) const;
    void                           make_the_cells              ( const std::function<void(const Cb &cb)> &f );
    template<int flags> void       make_lcs_from               ( const std::function<void( CP &, Dirac &dirac, int num_thread )> &cb, std::priority_queue<Msi> &base_queue, std::priority_queue<Msi> &queue, CP &lc, FinalCell *cell, const CpAndNum *path, TI path_len, int num_thread, N<flags>, const CP &starting_lc ) const;
    void                           make_znodes                 ( TZ *zcoords, TI *indices, const std::function<void(const Cb &cb)> &f, const SstLimits &sst );
    void                           display_vtk                 ( VtkOutput &vtk_output, BaseCell *cell, DisplayFlags display_flags ) const;
    const Dirac                   &get_dirac                   ( const DiracPn              &p , TI               ind  ) const { return p.diracs[ ind ]; }
    const Dirac                   &get_dirac                   ( const std::vector<DiracPn> &vp, std::pair<TI,TI> inds ) const { const DiracPn &p = vp[ inds.first ]; return p.diracs[ inds.second ]; }
    const Dirac                   &get_dirac                   ( int                           , const Dirac     &d    ) const { return d; }

    template<int a_n0,int f> void  cut_lc                      ( CP &lc, Pt c0, TF w0, FinalCell *dell, N<a_n0>, TI n0, N<f> ) const;


    // buffers
    std::vector<TZ>                znodes_keys;                ///< tmp znodes
    std::vector<TI>                znodes_inds;                ///< tmp indices for each znode ( ex: ppwns[ 0 ].positions[ ind.second ] to get positions )
    std::vector<std::pair<TI,TI>>  znodes_pnds;                ///< tmp indice pairs for each znode ( ex: ppwns[ ind.first ].positions[ ind.second ] to get positions )
    std::vector<Dirac>             znodes_vals;                ///< tmp positions and weights for each znode (for the case where we don't have all the positions)
    BumpPointerPool                mem_pool;                   ///< store the cells
    std::vector<std::size_t>       rs_tmps;                    ///< for the radix sort

    // result of cb calls
    bool                           use_dirac_pns;              ///<
    std::vector<DiracPn>           dirac_pns;                  ///< pointers to the dirac lists given by construct (if ptrs_survive_the_call)

    // sub structures

    // grid
    TF                             inv_step_length;
    TI                             nb_final_cells;
    TI                             nb_diracs_tot;
    TI                             nb_cb_calls;
    TF                             step_length;
    TF                             grid_length;
    std::vector<SstLimits>         sst_limits;                 ///<
    Pt                             min_point;
    Pt                             max_point;
    BaseCell                      *root_cell;
};

} // namespace sdot

#include "LGrid.tcc"

#endif // SDOT_LGrid_H
