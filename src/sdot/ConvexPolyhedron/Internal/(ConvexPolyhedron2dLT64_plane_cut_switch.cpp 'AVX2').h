#ifdef LOC_PARSE
#include "../../Support/SimdVec.h" 
#include <cstdint>
using TF = double;
using CI = std::uint64_t;
namespace sdot {
struct Nodes { TF xs[ 64 ], ys[ 64 ]; CI cut_ids[ 64 ]; };
struct S {
  void fu( S ) {}
  void f( Nodes nodes, int &num_cut, int nb_cuts, int nodes_size, TF **cut_dir, CI *cut_ids, TF *cut_ps ) {
#endif // LOC_PARSE
    using VF = SimdVec<TF,4>;
    using VC = SimdVec<CI,4>;
    VF px_0 = VF::load_aligned( nodes.xs + 0 );
    VF py_0 = VF::load_aligned( nodes.ys + 0 );
    VC pc_0 = VC::load_aligned( nodes.cut_ids + 0 );
    VF px_1 = VF::load_aligned( nodes.xs + 4 );
    VF py_1 = VF::load_aligned( nodes.ys + 4 );
    VC pc_1 = VC::load_aligned( nodes.cut_ids + 4 );
    for( ; ; ++num_cut ) {
        if ( num_cut == nb_cuts ) {
            VF::store_aligned( nodes.xs + 0, px_0 );
            VF::store_aligned( nodes.ys + 0, py_0 );
            VC::store_aligned( nodes.cut_ids + 0, pc_0 );
            VF::store_aligned( nodes.xs + 4, px_1 );
            VF::store_aligned( nodes.ys + 4, py_1 );
            VC::store_aligned( nodes.cut_ids + 4, pc_1 );
            return fu( *this );
        }
    
        // get distance and outside bit for each node
        std::uint16_t nmsk = 1 << nodes_size;
        VF cx = cut_dir[ 0 ][ num_cut ];
        VF cy = cut_dir[ 1 ][ num_cut ];
        VC ci = cut_ids[ num_cut ];
        VF cs = cut_ps[ num_cut ];
    
        VF bi_0 = px_0 * cx + py_0 * cy;
        VF bi_1 = px_1 * cx + py_1 * cy;
        std::uint16_t outside_nodes = ( ( ( bi_0 > cs ) << 0 ) | ( ( bi_1 > cs ) << 4 ) ) & ( nmsk - 1 );
        std::uint16_t case_code = outside_nodes | nmsk;
        VF di_0 = bi_0 - cs;
        VF di_1 = bi_1 - cs;
    
        // if nothing has changed => go to the next cut
        if ( outside_nodes == 0 )
            continue;
    
