#ifndef SDOT_CellBoundsPpos_H
#define SDOT_CellBoundsPpos_H

#include "../../Geometry/PointTraits.h"
#include <eigen3/Eigen/Cholesky>

namespace sdot {

/**
*/
template<class Pc>
class CellBoundsPpos {
public:
    static constexpr std::size_t w_bounds_order    = Pc::w_bounds_order; ///< poly order for weight bounding
    static constexpr std::size_t dim               = Pc::dim;            ///<
    using                        TF                = typename Pc::TF;
    using                        Pt                = typename PointTraits<TF,dim>::type;

    static constexpr int         nb_coeffs_w_bound = 1 + dim * ( w_bounds_order >= 1 ) + dim * ( dim + 1 ) / 2 * ( w_bounds_order >= 2 );
    using                        TMat              = Eigen::Matrix<TF,nb_coeffs_w_bound,nb_coeffs_w_bound>;
    using                        TVec              = Eigen::Matrix<TF,nb_coeffs_w_bound,1>;

    //        struct                     LocalSolver            {

    //            void                       clr                () { mat_weight = 0; vec_weight = 0; min_pos = + std::numeric_limits<TF>::max(); max_pos = - std::numeric_limits<TF>::max(); }
    //            void                       push               ( const LocalSolver &ls ) { using std::max; using std::min; max_weight = max( max_weight, ls.max_weight ); max_pos = max( max_pos, ls.max_pos ); min_pos = min( min_pos, ls.min_pos ); }
    //            void                       push               ( Pt pos, TF weight ) { using std::max; using std::min; max_weight = max( max_weight, weight ); max_pos = max( max_pos, pos ); min_pos = min( min_pos, pos ); }
    //            void                       store_to           ( BoundsD0 &bounds ) { bounds.max_weight = max_weight; bounds.min_pos = min_pos; bounds.max_pos = max_pos; }

    //            TMat                       mat_weight;
    //            TVec                       vec_weight;
    //            Pt                         min_pos;
    //            Pt                         max_pos;
    //        };

    TF                         get_w                  ( Pt pos ) const { TF res = poly_weight[ 0 ]; for( size_t d = 0; d < dim; ++d ) res += poly_weight[ d + 1 ] * pos[ d ]; return res; }

    TVec                       poly_weight;
    Pt                         min_pos;
    Pt                         max_pos;
};

} // namespace sdot

#include "CellBoundsPpos.tcc"

#endif // SDOT_CellBoundsPpos_H
