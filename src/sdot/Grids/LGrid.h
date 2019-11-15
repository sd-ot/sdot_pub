#ifndef SDOT_LGrid_H
#define SDOT_LGrid_H

#include "../Geometry/ConvexPolyhedronTraits.h"
#include "../Support/BumpPointerPool.h"
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

    // deduced types
    using                          CP                          = typename ConvexPolyhedronTraits<Pc>::type;
    using                          Cbd                         = std::function<void( const Dirac *diracs, TI nb_diracs, bool ptrs_survive_the_call )>; ///<
    using                          Cbw                         = std::function<void( const TF *weights, TI nb_diracs )>; ///<

    // flags for update_positions_and_weights, mod_weights, for_each_laguerre_cell
    enum {                         homogeneous_weights         = 1 };
    enum {                         ball_cut                    = 2 };

    // contruction methods
    /* ctor */                     LGrid                       ( std::size_t max_diracs_per_cell = 11 );

    // grid makers. In all the cases, diracs are copied.
    template<int flags> void       update                      ( const Dirac *diracs, TI nb_diracs, N<flags> );
    template<int flags> void       update                      ( const std::function<void( const Cbd &cb )> &f, N<flags> );

    // traversal/information
    template<int flags> int        for_each_laguerre_cell      ( const std::function<void( CP &lc, int num_thread )> &f, const CP &starting_lc, N<flags>, bool stop_if_void_lc = false ); ///< version with num_thread
    TI                             nb_diracs                   () const { return nb_diracs_tot; }

    // display
    void                           write_to_stream             ( std::ostream &os ) const;
    void                           display_tikz                ( std::ostream &os, TF scale = 1.0 ) const;
    void                           display_vtk                 ( VtkOutput &vtk_output, int disp_weights = 0 ) const; ///< for debug purpose

    // values used by update
    TI                             max_diracs_per_cell;
    TI                             max_diracs_per_sst;
    std::vector<Pt>                translations;

private:
    static constexpr int           nb_bits_per_axis            = 20;
    static constexpr int           sizeof_zcoords              = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    using                          CellBounds                  = typename CellBoundsTraits<Pc>::type;
    using                          LocalSolver                 = typename CellBounds::LocalSolver;
    using                          FinalCell                   = LGridFinalCell<Pc>;
    using                          SuperCell                   = LGridSuperCell<Pc>;
    using                          BaseCell                    = LGridBaseCell<Pc>;
    using                          TZ                          = std::uint64_t; ///< zcoords

    struct                         TmpLevelInfo                {
        void                       clr                         () { num_sub_cell = 0; nb_sub_cells = 0; ls.clr(); }
        BaseCell                  *sub_cells[ 1 << dim ];      ///<
        TI                         num_sub_cell;               ///<
        TI                         nb_sub_cells;               ///<
        LocalSolver                ls;
    };

    struct                         SstLimits                   { TZ beg_zcoords, end_zcoords; TI nb_diracs; int level; };
    struct                         CpAndNum                    { const SuperCell *cell; TI num; };
    struct                         Msi                         { bool operator<( const Msi &that ) const { return dist > that.dist; } Pt center; const BaseCell *cell; TF dist; };

    template<int flags> void       compute_grid_dims_and_ppwns ( const std::function<void(const Cbd &cb)> &f, N<flags> );
    void                           update_cell_bounds_phase_1  ( BaseCell *cell, BaseCell **path, int level );
    void                           fill_grid_using_zcoords     ( const Pt *positions, const TF *weights, TI nb_diracs );
    void                           make_znodes_with_1ppwn_ssst ( const SstLimits &sst, const Pt *positions, TI nb_diracs ); ///< several sst case
    void                           make_znodes_with_1ppwn_1sst ( const Pt *positions, TI nb_diracs );                       ///< only one sst case (=> no need to make a test)
    template<int flags> void       compute_sst_limits          ( const std::function<void(const Cbd &cb)> &f, N<flags> );
    template<class Ps> void        make_the_cells_for          ( const SstLimits &sst, TmpLevelInfo *level_info, Ps ps );
    void                           write_to_stream             ( std::ostream &os, BaseCell *cell, std::string sp ) const;
    void                           mod_weights_rec             ( const std::function<void( const Pt &position, TF &weight, Af &af )> &f, BaseCell *cell, LocalSolver *local_solvers, int level );
    TI                            *znodes_seconds              ( const Ppwn & ) { return znodes_inds.data(); }
    std::pair<TI,TI>              *znodes_seconds              ( const std::vector<Ppwn> & ) { return znodes_pnds.data(); }
    Pwi                           *znodes_seconds              ( int ) { return znodes_vals.data(); }
    template<int flags> bool       can_be_evicted              ( const CP &lc, Pt &c0, TF w0, const CellBoundsP0<Pc> &bounds, N<flags> ) const;
    template<int flags> bool       can_be_evicted              ( const CP &lc, Pt &c0, TF w0, const CellBoundsPpos<Pc> &bounds, N<flags> ) const;
    void                           make_the_cells              ( const std::function<void(const Cbd &cb)> &f );
    template<int flags> void       make_lcs_from               ( const std::function<void( CP &, int num_thread )> &cb, std::priority_queue<Msi> &base_queue, std::priority_queue<Msi> &queue, CP &lc, const FinalCell *cell, const CpAndNum *path, TI path_len, int num_thread, N<flags>, const CP &starting_lc ) const;
    void                           make_znodes                 ( TZ *zcoords, TI *indices, const std::function<void(const Cbd &cb)> &f, const SstLimits &sst );
    Pwi                            get_pwi                     ( const Ppwn &p, TI ind ) const { return { p.positions[ ind ], p.weights[ ind ], ind }; }
    Pwi                            get_pwi                     ( const std::vector<Ppwn> &ppwns, std::pair<TI,TI> inds ) const { const Ppwn &p = ppwns[ inds.first ]; return { p.positions[ inds.second ], p.weights[ inds.second ], p.off_diracs + inds.second }; }
    Pwi                            get_pwi                     ( int, const Pwi &pwi ) const { return pwi; }
    void                           display_vtk                     ( VtkOutput &vtk_output, BaseCell *cell, int disp_weights ) const;

    template<int a_n0,int f> void  cut_lc                      ( CP &lc, Pt c0, TF w0, const FinalCell *dell, N<a_n0>, TI n0, N<f> ) const;


    // buffers
    std::vector<TZ>                znodes_keys;                ///< tmp znodes
    std::vector<TI>                znodes_inds;                ///< tmp indices for each znode ( ex: ppwns[ 0 ].positions[ ind.second ] to get positions )
    std::vector<std::pair<TI,TI>>  znodes_pnds;                ///< tmp indice pairs for each znode ( ex: ppwns[ ind.first ].positions[ ind.second ] to get positions )
    std::vector<Pwi>               znodes_vals;                ///< tmp positions and weights for each znode (for the case where we don't have all the positions)
    BumpPointerPool                mem_pool;                   ///< store the cells
    std::vector<std::size_t>       rs_tmps;                    ///< for the radix sort

    // result of cb calls
    bool                           use_ppwns;                  ///<
    std::vector<Ppwn>              ppwns;                      ///< nb_diracs + pointers to positions and weights

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
