#ifndef SDOT_CellBoundsPpos_H
#define SDOT_CellBoundsPpos_H

#include "../../Support/PointTraits.h"
#include <eigen3/Eigen/Cholesky>

namespace sdot {

/**
*/
template<class Pc>
class CellBoundsPpos {
public:
    static constexpr std::size_t w_bounds_order    = Pc::w_bounds_order; ///< poly order for weight bounding
    static constexpr bool        need_phase_1      = true;
    static constexpr std::size_t dim               = Pc::dim;            ///<
    using                        TF                = typename Pc::TF;
    using                        Pt                = typename PointTraits<TF,dim>::type;

    static constexpr int         nb_coeffs_w_bound = 1 + dim * ( w_bounds_order >= 1 ) + dim * ( dim + 1 ) / 2 * ( w_bounds_order >= 2 );
    using                        TMat              = Eigen::Matrix<TF,nb_coeffs_w_bound,nb_coeffs_w_bound>;
    using                        TVec              = Eigen::Matrix<TF,nb_coeffs_w_bound,1>;
    using                        TPol              = std::array<TF,nb_coeffs_w_bound>;

    struct                       LocalSolver       {
        void                     clr               ();
        void                     push              ( Pt pos, TF weight );
        void                     push              ( const LocalSolver &ls );
        void                     store_to          ( CellBoundsPpos &bounds );

        TMat                     mat_weight;
        TVec                     vec_weight;
        Pt                       min_pos;
        Pt                       max_pos;
    };

    void                         push                   ( Pt pos, TF weight );
    TF                           get_w                  ( Pt pos ) const;

    TPol                         poly_weight;
    Pt                           min_pos;
    Pt                           max_pos;
};

} // namespace sdot

#include "CellBoundsPpos.tcc"

#endif // SDOT_CellBoundsPpos_H
