#ifdef __AVX2__
#include "../../Support/SimdVec.h"
void ConvexPolyhedron2dLt64_cut( double *px, double *py, std::size_t *pi, int &nodes_size, const double *cut_x, const double *cut_y, const double *cut_s, const std::size_t *cut_i, int cn ) {
    using namespace sdot;
    using TF = double;
    using TC = std::size_t;
    using VF = SimdVec<TF,4>;
    using VC = SimdVec<TC,4>;
    VF px_0 = VF::load_aligned( px + 0 );
    VF py_0 = VF::load_aligned( py + 0 );
    VC pi_0 = VC::load_aligned( pi + 0 );
    for( int num_cut = 0; num_cut < cn; ++num_cut ) {
        int nmsk = 1 << nodes_size;
        VF cx = cut_x[ num_cut ];
        VF cy = cut_y[ num_cut ];
        VF cs = cut_s[ num_cut ];
        
        VF bi_0 = px_0 * cx + py_0 * cy;
        int outside_nodes = ( ( bi_0 > cs ) << 0 );
        int case_code = ( outside_nodes & ( nmsk - 1 ) ) | nmsk;
        VF di_0 = bi_0 - cs;
        
        // if nothing has changed => go to the next cut
        if ( outside_nodes == 0 )
            continue;
        static void *dispatch_table[] = { &&case_not_handled, &&case_not_handled, &&case_not_handled, &&case_not_handled, &&case_not_handled, &&case_not_handled, &&case_not_handled, &&case_not_handled, &&case_1, &&case_2, &&case_3, &&case_4, &&case_5, &&case_6, &&case_7, &&case_8,  };
        goto *dispatch_table[ case_code ];
        case_not_handled: {
            break;
        }
        case_1: {
            // everything is inside
            continue;
        }
        case_8: {
            // everything is outside
            continue;
        }
        case_3: {
            // mod=0,[0,1],[1,2],2, swith_cuts=0
            auto R0 = di_0[ 2 ];
            auto R1 = di_0[ 0 ];
            SimdVec<TF,2> R2{ R1, R0 };
            auto R3 = di_0[ 1 ];
            SimdVec<TF,2> R4{ R3, R3 };
            auto R5 = R2 - R4;
            auto R6 = R4 / R5;
            auto R7 = px_0[ 2 ];
            auto R8 = px_0[ 1 ];
            SimdVec<TF,2> R9{ R8, R8 };
            auto R10 = px_0[ 0 ];
            SimdVec<TF,2> R11{ R10, R7 };
            auto R12 = R9 - R11;
            auto R13 = R6 * R12;
            auto R14 = R9 + R13;
            auto R15 = R14[ 1 ];
            auto R16 = R14[ 0 ];
            SimdVec<TF,4> R17{ R10, R16, R15, R7 };
            px_0 = R17;
            continue;
        }
        case_2: {
            // mod=[0,1],1,2,[2,0], swith_cuts=0
            auto R0 = di_0[ 1 ];
            auto R1 = di_0[ 2 ];
            auto R2 = di_0[ 0 ];
            SimdVec<TF,2> R3{ R0, R2 };
            SimdVec<TF,2> R4{ R2, R1 };
            auto R5 = R3 - R4;
            auto R6 = R4 / R5;
            auto R7 = px_0[ 1 ];
            auto R8 = px_0[ 2 ];
            auto R9 = px_0[ 0 ];
            SimdVec<TF,2> R10{ R7, R9 };
            SimdVec<TF,2> R11{ R9, R8 };
            auto R12 = R11 - R10;
            auto R13 = R6 * R12;
            auto R14 = R11 + R13;
            auto R15 = R14[ 1 ];
            auto R16 = R14[ 0 ];
            SimdVec<TF,4> R17{ R16, R7, R8, R15 };
            px_0 = R17;
            continue;
        }
        case_6: {
            // mod=[0,1],1,[1,2], swith_cuts=0
            auto R0 = di_0[ 2 ];
            auto R1 = di_0[ 1 ];
            SimdVec<TF,2> R2{ R1, R0 };
            auto R3 = di_0[ 0 ];
            SimdVec<TF,2> R4{ R3, R1 };
            auto R5 = R2 - R4;
            auto R6 = R4 / R5;
            auto R7 = px_0[ 2 ];
            auto R8 = px_0[ 1 ];
            SimdVec<TF,2> R9{ R8, R7 };
            auto R10 = px_0[ 0 ];
            SimdVec<TF,2> R11{ R10, R8 };
            auto R12 = R11 - R9;
            auto R13 = R6 * R12;
            auto R14 = R11 + R13;
            auto R15 = R14[ 1 ];
            auto R16 = R14[ 0 ];
            SimdVec<TF,4> R17{ R16, R8, R15, R15 };
            px_0 = R17;
            continue;
        }
        case_4: {
            // mod=[1,2],2,[2,0], swith_cuts=0
            auto R0 = di_0[ 2 ];
            SimdVec<TF,2> R1{ R0, R0 };
            auto R2 = di_0[ 0 ];
            auto R3 = di_0[ 1 ];
            SimdVec<TF,2> R4{ R3, R2 };
            auto R5 = R1 - R4;
            auto R6 = R4 / R5;
            auto R7 = px_0[ 2 ];
            SimdVec<TF,2> R8{ R7, R7 };
            auto R9 = px_0[ 0 ];
            auto R10 = px_0[ 1 ];
            SimdVec<TF,2> R11{ R10, R9 };
            auto R12 = R11 - R8;
            auto R13 = R6 * R12;
            auto R14 = R11 + R13;
            auto R15 = R14[ 1 ];
            auto R16 = R14[ 0 ];
            SimdVec<TF,4> R17{ R16, R7, R15, R15 };
            px_0 = R17;
            continue;
        }
        case_5: {
            // mod=[2,0],0,1,[1,2], swith_cuts=0
            auto R0 = di_0[ 0 ];
            auto R1 = di_0[ 1 ];
            auto R2 = di_0[ 2 ];
            SimdVec<TF,2> R3{ R0, R2 };
            SimdVec<TF,2> R4{ R2, R1 };
            auto R5 = R3 - R4;
            auto R6 = R4 / R5;
            auto R7 = px_0[ 0 ];
            auto R8 = px_0[ 1 ];
            auto R9 = px_0[ 2 ];
            SimdVec<TF,2> R10{ R7, R9 };
            SimdVec<TF,2> R11{ R9, R8 };
            auto R12 = R11 - R10;
            auto R13 = R6 * R12;
            auto R14 = R11 + R13;
            auto R15 = R14[ 1 ];
            auto R16 = R14[ 0 ];
            SimdVec<TF,4> R17{ R16, R7, R8, R15 };
            px_0 = R17;
            continue;
        }
        case_7: {
            // mod=[2,0],0,[0,1], swith_cuts=0
            auto R0 = di_0[ 1 ];
            auto R1 = di_0[ 0 ];
            SimdVec<TF,2> R2{ R1, R0 };
            auto R3 = di_0[ 2 ];
            SimdVec<TF,2> R4{ R3, R1 };
            auto R5 = R2 - R4;
            auto R6 = R4 / R5;
            auto R7 = px_0[ 1 ];
            auto R8 = px_0[ 0 ];
            SimdVec<TF,2> R9{ R8, R7 };
            auto R10 = px_0[ 2 ];
            SimdVec<TF,2> R11{ R10, R8 };
            auto R12 = R11 - R9;
            auto R13 = R6 * R12;
            auto R14 = R11 + R13;
            auto R15 = R14[ 1 ];
            auto R16 = R14[ 0 ];
            SimdVec<TF,4> R17{ R16, R8, R15, R15 };
            px_0 = R17;
            continue;
        }
    }
    VF::store_aligned( px + 0, px_0 );
    VF::store_aligned( py + 0, py_0 );
    VC::store_aligned( pi + 0, pi_0 );
}
#endif // __AVX2__
