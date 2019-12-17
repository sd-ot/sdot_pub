#include "../../src/sdot/Support/SimdVec.h"
#include <iostream>
#include <chrono>

//// nsmake cxx_name clang++
//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -ffast-math
//// nsmake cpp_flag -O3

using CI = std::uint64_t;
using TF = double;

void __attribute__ ((noinline)) cut_proc( std::size_t nb_reps, TF *xs, TF *ys, TF *di, CI *cs ) {
    using namespace sdot;
    using VF = SimdVec<TF,4>;
    using VC = SimdVec<CI,4>;

    VF px_0 = VF::load_aligned( xs + 0 );
    VF py_0 = VF::load_aligned( ys + 0 );
    //VC pc_0 = VC::load_aligned( cs + 0 );
    VF px_1 = VF::load_aligned( xs + 4 );
    VF py_1 = VF::load_aligned( ys + 4 );
    //VC pc_1 = VC::load_aligned( cs + 4 );

    for( std::size_t rep = 0; rep < nb_reps; ++rep ) {
        VF di_0 = VF::load_aligned( di + 0 ) + VF( rep );
        VF di_1 = VF::load_aligned( di + 4 ) + VF( rep );

        //        // 0.780245
        //        // nb_nodes:5 outside:00000011 ops:[4,0],[1,2],2,3,4
        //        SimdVec<TF,2> di_a( di_1[ 0 ], di_0[ 1 ] );
        //        SimdVec<TF,2> di_b( di_0[ 0 ], di_0[ 2 ] );
        //        SimdVec<TF,2> px_a( px_1[ 0 ], px_0[ 1 ] );
        //        SimdVec<TF,2> px_b( px_0[ 0 ], px_0[ 2 ] );
        //        SimdVec<TF,2> py_a( py_1[ 0 ], py_0[ 1 ] );
        //        SimdVec<TF,2> py_b( py_0[ 0 ], py_0[ 2 ] );
        //        SimdVec<TF,2> dm_s = di_a / ( di_b - di_a );
        //        SimdVec<TF,2> nx_s = px_a - dm_s * ( px_b - px_a );
        //        SimdVec<TF,2> ny_s = py_a - dm_s * ( py_b - py_a );
        //        px_0[ 0 ] = nx_s[ 0 ];
        //        px_0[ 1 ] = nx_s[ 1 ];
        //        py_0[ 0 ] = ny_s[ 0 ];
        //        py_0[ 1 ] = ny_s[ 1 ];

        TF dm_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
        TF dm_1 = di_0[ 1 ] / ( di_0[ 2 ] - di_0[ 1 ] );
        TF nx_0 = px_1[ 0 ] - dm_0 * ( px_0[ 0 ] - px_1[ 0 ] );
        TF ny_0 = py_1[ 0 ] - dm_0 * ( py_0[ 0 ] - py_1[ 0 ] );
        TF nx_1 = px_0[ 1 ] - dm_1 * ( px_0[ 2 ] - px_0[ 1 ] );
        TF ny_1 = py_0[ 1 ] - dm_1 * ( py_0[ 2 ] - py_0[ 1 ] );
        px_0[ 0 ] = nx_0;
        py_0[ 0 ] = ny_0;
        px_0[ 1 ] = nx_1;
        py_0[ 1 ] = ny_1;
    }

    VF::store_aligned( xs + 0, px_0 );
    VF::store_aligned( ys + 0, py_0 );
    VF::store_aligned( xs + 4, px_1 );
    VF::store_aligned( ys + 4, py_1 );
}

int main( int /*argc*/, char **/*argv*/ ) {
    alignas( 64 ) TF xs[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    alignas( 64 ) TF ys[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    alignas( 64 ) TF di[] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    alignas( 64 ) CI cs[] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    auto t0 = std::chrono::high_resolution_clock::now();
    cut_proc( 200000000, xs, ys, di, cs );
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>( t1 - t0 ).count() / 1e6 << std::endl;
}

//401200:       62 f1 a7 08 7b f0       vcvtusi2sd %rax,%xmm11,%xmm6
//401206:       c4 e2 7d 19 fe          vbroadcastsd %xmm6,%ymm7
//40120b:       c5 bd 58 ff             vaddpd %ymm7,%ymm8,%ymm7
//40120f:       c4 e3 79 05 e7 01       vpermilpd $0x1,%xmm7,%xmm4
//401215:       c4 e3 7d 19 fd 01       vextractf128 $0x1,%ymm7,%xmm5
//40121b:       c5 d3 5c ec             vsubsd %xmm4,%xmm5,%xmm5
//40121f:       c5 db 5e e5             vdivsd %xmm5,%xmm4,%xmm4
//401223:       c5 b3 58 ee             vaddsd %xmm6,%xmm9,%xmm5
//401227:       c5 c3 5c f5             vsubsd %xmm5,%xmm7,%xmm6
//40122b:       c5 d3 5e ee             vdivsd %xmm6,%xmm5,%xmm5
//40122f:       c4 e3 79 05 f1 01       vpermilpd $0x1,%xmm1,%xmm6
//401235:       c4 e3 7d 19 cf 01       vextractf128 $0x1,%ymm1,%xmm7
//40123b:       c5 cb 5c ff             vsubsd %xmm7,%xmm6,%xmm7
//40123f:       c4 e2 d9 a9 fe          vfmadd213sd %xmm6,%xmm4,%xmm7
//401244:       c4 e3 79 05 f3 01       vpermilpd $0x1,%xmm3,%xmm6
//40124a:       c4 e3 7d 19 d8 01       vextractf128 $0x1,%ymm3,%xmm0
//401250:       c5 cb 5c c0             vsubsd %xmm0,%xmm6,%xmm0
//401254:       c4 e2 d9 a9 c6          vfmadd213sd %xmm6,%xmm4,%xmm0
//401259:       c5 ab 5c e1             vsubsd %xmm1,%xmm10,%xmm4
//40125d:       c4 c2 d1 a9 e2          vfmadd213sd %xmm10,%xmm5,%xmm4
//401262:       c5 c1 14 e4             vunpcklpd %xmm4,%xmm7,%xmm4
//401266:       c5 eb 5c f3             vsubsd %xmm3,%xmm2,%xmm6
//40126a:       c4 e2 d1 a9 f2          vfmadd213sd %xmm2,%xmm5,%xmm6
//40126f:       c5 f9 14 c6             vunpcklpd %xmm6,%xmm0,%xmm0
//401273:       c4 e3 75 18 cc 01       vinsertf128 $0x1,%xmm4,%ymm1,%ymm1
//401279:       c4 e3 65 18 d8 01       vinsertf128 $0x1,%xmm0,%ymm3,%ymm3
