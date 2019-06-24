#ifndef SDOT_ZGRID_H
#define SDOT_ZGRID_H

#include "Geometry/ConvexPolyhedron2.h"

namespace sdot {

/**
*/
template<class Pc>
class ZGrid {
public:
    static constexpr std::size_t dim                    = Pc::dim;         ///<
    using                        CP2                    = ConvexPolyhedron2<Pc>;
    using                        CP3                    = ConvexPolyhedron2<Pc>;
    using                        CP                     = typename std::conditional<dim==3,CP3,CP2>::type;

    using                        TF                     = typename CP::TF; ///< floating point type
    using                        TI                     = typename CP::TI; ///< index type
    using                        CI                     = typename CP::CI; ///< cut info
    using                        Pt                     = typename CP::Pt; ///< cut info

    /* ctor */                   ZGrid                  ( std::size_t max_diracs_per_cell = 11 );

    void                         update                 ( const Pt *positions, const TF *weights, std::size_t nb_diracs, bool positions_have_changed = true, bool weights_have_changed = true, bool ball_cut = false );
    template<int bc> void        update                 ( const Pt *positions, const TF *weights, std::size_t nb_diracs, bool positions_have_changed, bool weights_have_changed, N<bc> ball_cut );

    int                          for_each_laguerre_cell ( const std::function<void( CP &lc, TI num, int num_thread )> &f, const CP &starting_lc, const Pt *positions, const TF *weights, TI nb_diracs, bool stop_if_void_lc = false, bool ball_cut = false ); ///< version with num_thread
    template<int bc> int         for_each_laguerre_cell ( const std::function<void( CP &lc, TI num, int num_thread )> &f, const CP &starting_lc, const Pt *positions, const TF *weights, TI nb_diracs, bool stop_if_void_lc, N<bc> ball_cut ); ///< version with num_thread

    void                         display                ( VtkOutput &vtk_output ) const; ///< for debug purpose

    // values used by update
    int                          max_diracs_per_cell;
    std::vector<Pt>              translations;

private:
    static constexpr int         nb_bits_per_axis       = 63 / dim;
    static constexpr int         sizeof_zcoords         = ( dim * nb_bits_per_axis + 7 ) / 8; ///< nb meaningful bytes in z-coordinates
    using                        TZ                     = std::uint64_t; ///< zcoords

    struct                       Cell {
        TI                       dpc_offset;            ///< offsets in grid.dpc_values
        TZ                       zcoords;
        TF                       size;
        Pt                       pos;
    };

    struct                       Grid {
        std::vector<TI>          cell_index_vs_dirac_number; ///< num cell for each dirac
        std::vector<TI>          dirac_indices;              ///< filled only if blocks.size() > 1
        std::vector<TI>          dpc_values;                 ///< diracs indices for each cell
        std::vector<TI>          ng_indices;                 ///< list of cell index of direct neighbors
        std::vector<TI>          ng_offsets;                 ///< offsets in ng_indices for each cell
        TF                       min_weight;
        TF                       max_weight;
        std::vector<Cell>        cells;
    };

    struct                       ZNode {
        void                     write_to_stream        ( std::ostream &os ) const { os << zcoords << "[" << index << "]"; }
        TZ                       operator>>             ( int shift ) const { return zcoords >> shift; }
        TZ                       zcoords; ///<
        TI                       index;
    };

    void                         fill_grid_using_zcoords( const Pt *positions, const TF *weights, std::size_t nb_diracs );
    void                         repl_zcoords_by_ccoords( const TF *weights );
    void                         find_englobing_cousins ( const Pt *positions ); ///< find englobing cells for each dirac (and for each grid). Must be done after repl_zcoords_by_ccoords
    void                         update_the_limits      ( const Pt *positions, const TF *weights, std::size_t nb_diracs );
    void                         update_neighbors       ();
    void                         fill_the_grid          ( const Pt *positions, const TF *weights, std::size_t nb_diracs );
    template<int bc> TF          min_w_to_cut           ( const CP &lc, Pt c0, TF w0, const Cell &cr_cell, const Pt *positions, const TF *weights, N<bc> );
    template<class C> TZ         zcoords_for            ( const C &pos ); ///< floating point position
    template<int d>   TZ         ng_zcoord              ( TZ zcoords, TZ off, N<d> ) const;

    // tmp
    using                        RsTmp                  = std::vector<std::array<std::size_t,256>>;
    RsTmp                        rs_tmps;
    std::vector<ZNode>           znodes;                ///< tmp znodes
    std::vector<ZNode>           zcells;                ///<

    //
    TF                           step_length;
    TF                           grid_length;
    TF                           min_weight;
    TF                           max_weight;
    Pt                           min_point;
    Pt                           max_point;
    Grid                         grid;                  ///<
};

} // namespace sdot

#include "ZGrid.tcc"

#endif // SDOT_ZGRID_H
