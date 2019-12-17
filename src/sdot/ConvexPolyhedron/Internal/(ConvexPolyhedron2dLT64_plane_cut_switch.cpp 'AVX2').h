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
    
        static void *dispatch_table[] = {
            &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_1, &&case_2, &&case_3, &&case_4, &&case_5, &&case_6, &&case_7, &&case_8,
            &&case_1, &&case_9, &&case_10, &&case_11, &&case_12, &&case_0, &&case_13, &&case_14, &&case_15, &&case_16, &&case_0, &&case_17, &&case_18, &&case_19, &&case_20, &&case_8,
            &&case_1, &&case_21, &&case_22, &&case_23, &&case_24, &&case_0, &&case_25, &&case_26, &&case_27, &&case_0, &&case_0, &&case_0, &&case_28, &&case_0, &&case_29, &&case_30,
            &&case_31, &&case_32, &&case_0, &&case_33, &&case_0, &&case_0, &&case_0, &&case_34, &&case_35, &&case_36, &&case_0, &&case_37, &&case_38, &&case_39, &&case_40, &&case_8,
            &&case_1, &&case_41, &&case_42, &&case_43, &&case_44, &&case_0, &&case_45, &&case_46, &&case_47, &&case_0, &&case_0, &&case_0, &&case_48, &&case_0, &&case_49, &&case_50,
            &&case_51, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_52, &&case_0, &&case_0, &&case_0, &&case_53, &&case_0, &&case_54, &&case_55,
            &&case_56, &&case_57, &&case_0, &&case_58, &&case_0, &&case_0, &&case_0, &&case_59, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_60,
            &&case_61, &&case_62, &&case_0, &&case_63, &&case_0, &&case_0, &&case_0, &&case_64, &&case_65, &&case_66, &&case_0, &&case_67, &&case_68, &&case_69, &&case_70, &&case_8,
            &&case_1, &&case_71, &&case_72, &&case_73, &&case_74, &&case_0, &&case_75, &&case_76, &&case_77, &&case_0, &&case_0, &&case_0, &&case_78, &&case_0, &&case_79, &&case_80,
            &&case_81, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_82, &&case_0, &&case_0, &&case_0, &&case_83, &&case_0, &&case_84, &&case_85,
            &&case_86, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0,
            &&case_87, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_88, &&case_0, &&case_0, &&case_0, &&case_89, &&case_0, &&case_90, &&case_91,
            &&case_92, &&case_93, &&case_0, &&case_94, &&case_0, &&case_0, &&case_0, &&case_95, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_96,
            &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_97,
            &&case_98, &&case_99, &&case_0, &&case_100, &&case_0, &&case_0, &&case_0, &&case_101, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_102,
            &&case_103, &&case_104, &&case_0, &&case_105, &&case_0, &&case_0, &&case_0, &&case_106, &&case_107, &&case_108, &&case_0, &&case_109, &&case_110, &&case_111, &&case_112, &&case_8,
            &&case_1, &&case_113, &&case_114, &&case_115, &&case_116, &&case_0, &&case_117, &&case_118, &&case_119, &&case_0, &&case_0, &&case_0, &&case_120, &&case_0, &&case_121, &&case_122,
            &&case_123, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_124, &&case_0, &&case_0, &&case_0, &&case_125, &&case_0, &&case_126, &&case_127,
            &&case_128, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0,
            &&case_129, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_130, &&case_0, &&case_0, &&case_0, &&case_131, &&case_0, &&case_132, &&case_133,
            &&case_134, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0,
            &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0,
            &&case_135, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0,
            &&case_136, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_137, &&case_0, &&case_0, &&case_0, &&case_138, &&case_0, &&case_139, &&case_140,
            &&case_141, &&case_142, &&case_0, &&case_143, &&case_0, &&case_0, &&case_0, &&case_144, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_145,
            &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_146,
            &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0,
            &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_147,
            &&case_148, &&case_149, &&case_0, &&case_150, &&case_0, &&case_0, &&case_0, &&case_151, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_152,
            &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_153,
            &&case_154, &&case_155, &&case_0, &&case_156, &&case_0, &&case_0, &&case_0, &&case_157, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_0, &&case_158,
            &&case_159, &&case_160, &&case_0, &&case_161, &&case_0, &&case_0, &&case_0, &&case_162, &&case_163, &&case_164, &&case_0, &&case_165, &&case_166, &&case_167, &&case_168, &&case_8,
        };
        goto *dispatch_table[ case_code ];
      case_1: {
        // everything is inside
        continue;
      }
      case_8: {
        // everything is outside
        nodes_size = 0;
        return fu( *this );
      }
      case_2: {
        // nb_nodes:3 outside:00000001 ops:[0,1],1,2,[2,0]
      }
      case_3: {
        // nb_nodes:3 outside:00000010 ops:0,[0,1],[1,2],2
      }
      case_4: {
        // nb_nodes:3 outside:00000011 ops:[2,0],[1,2],2
      }
      case_5: {
        // nb_nodes:3 outside:00000100 ops:0,1,[1,2],[2,0]
      }
      case_6: {
        // nb_nodes:3 outside:00000101 ops:[0,1],1,[1,2]
      }
      case_7: {
        // nb_nodes:3 outside:00000110 ops:0,[0,1],[2,0]
      }
      case_9: {
        // nb_nodes:4 outside:00000001 ops:[0,1],1,2,3,[3,0]
      }
      case_10: {
        // nb_nodes:4 outside:00000010 ops:[0,1],[1,2],2,3,0
      }
      case_11: {
        // nb_nodes:4 outside:00000011 ops:[3,0],[1,2],2,3
      }
      case_12: {
        // nb_nodes:4 outside:00000100 ops:0,1,[1,2],[2,3],3
      }
      case_13: {
        // nb_nodes:4 outside:00000110 ops:0,[0,1],[2,3],3
      }
      case_14: {
        // nb_nodes:4 outside:00000111 ops:[2,3],3,[3,0]
      }
      case_15: {
        // nb_nodes:4 outside:00001000 ops:0,1,2,[2,3],[3,0]
      }
      case_16: {
        // nb_nodes:4 outside:00001001 ops:[0,1],1,2,[2,3]
      }
      case_17: {
        // nb_nodes:4 outside:00001011 ops:[2,3],[1,2],2
      }
      case_18: {
        // nb_nodes:4 outside:00001100 ops:0,1,[1,2],[3,0]
      }
      case_19: {
        // nb_nodes:4 outside:00001101 ops:[0,1],1,[1,2]
      }
      case_20: {
        // nb_nodes:4 outside:00001110 ops:0,[0,1],[3,0]
      }
      case_21: {
        // nb_nodes:5 outside:00000001 ops:[0,1],1,2,3,4,[4,0]
      }
      case_22: {
        // nb_nodes:5 outside:00000010 ops:[0,1],[1,2],2,3,4,0
      }
      case_23: {
        // nb_nodes:5 outside:00000011 ops:[4,0],[1,2],2,3,4
      }
      case_24: {
        // nb_nodes:5 outside:00000100 ops:0,1,[1,2],[2,3],3,4
      }
      case_25: {
        // nb_nodes:5 outside:00000110 ops:0,[0,1],[2,3],3,4
      }
      case_26: {
        // nb_nodes:5 outside:00000111 ops:4,[4,0],[2,3],3
      }
      case_27: {
        // nb_nodes:5 outside:00001000 ops:0,1,2,[2,3],[3,4],4
      }
      case_28: {
        // nb_nodes:5 outside:00001100 ops:0,1,[1,2],[3,4],4
      }
      case_29: {
        // nb_nodes:5 outside:00001110 ops:0,[0,1],[3,4],4
      }
      case_30: {
        // nb_nodes:5 outside:00001111 ops:[3,4],4,[4,0]
      }
      case_31: {
        // nb_nodes:5 outside:00010000 ops:0,1,2,3,[3,4],[4,0]
      }
      case_32: {
        // nb_nodes:5 outside:00010001 ops:[0,1],1,2,3,[3,4]
      }
      case_33: {
        // nb_nodes:5 outside:00010011 ops:[3,4],[1,2],2,3
      }
      case_34: {
        // nb_nodes:5 outside:00010111 ops:[2,3],3,[3,4]
      }
      case_35: {
        // nb_nodes:5 outside:00011000 ops:0,1,2,[2,3],[4,0]
      }
      case_36: {
        // nb_nodes:5 outside:00011001 ops:[0,1],1,2,[2,3]
      }
      case_37: {
        // nb_nodes:5 outside:00011011 ops:[2,3],[1,2],2
      }
      case_38: {
        // nb_nodes:5 outside:00011100 ops:0,1,[1,2],[4,0]
      }
      case_39: {
        // nb_nodes:5 outside:00011101 ops:[0,1],1,[1,2]
      }
      case_40: {
        // nb_nodes:5 outside:00011110 ops:0,[0,1],[4,0]
      }
      case_41: {
        // nb_nodes:6 outside:00000001 ops:[0,1],1,2,3,4,5,[5,0]
      }
      case_42: {
        // nb_nodes:6 outside:00000010 ops:[0,1],[1,2],2,3,4,5,0
      }
      case_43: {
        // nb_nodes:6 outside:00000011 ops:[5,0],[1,2],2,3,4,5
      }
      case_44: {
        // nb_nodes:6 outside:00000100 ops:1,[1,2],[2,3],3,4,5,0
      }
      case_45: {
        // nb_nodes:6 outside:00000110 ops:0,[0,1],[2,3],3,4,5
      }
      case_46: {
        // nb_nodes:6 outside:00000111 ops:5,[5,0],[2,3],3,4
      }
      case_47: {
        // nb_nodes:6 outside:00001000 ops:0,1,2,[2,3],[3,4],4,5
      }
      case_48: {
        // nb_nodes:6 outside:00001100 ops:0,1,[1,2],[3,4],4,5
      }
      case_49: {
        // nb_nodes:6 outside:00001110 ops:0,[0,1],[3,4],4,5
      }
      case_50: {
        // nb_nodes:6 outside:00001111 ops:[3,4],4,5,[5,0]
      }
      case_51: {
        // nb_nodes:6 outside:00010000 ops:0,1,2,3,[3,4],[4,5],5
      }
      case_52: {
        // nb_nodes:6 outside:00011000 ops:0,1,2,[2,3],[4,5],5
      }
      case_53: {
        // nb_nodes:6 outside:00011100 ops:0,1,[1,2],[4,5],5
      }
      case_54: {
        // nb_nodes:6 outside:00011110 ops:0,[0,1],[4,5],5
      }
      case_55: {
        // nb_nodes:6 outside:00011111 ops:[4,5],5,[5,0]
      }
      case_56: {
        // nb_nodes:6 outside:00100000 ops:0,1,2,3,4,[4,5],[5,0]
      }
      case_57: {
        // nb_nodes:6 outside:00100001 ops:[0,1],1,2,3,4,[4,5]
      }
      case_58: {
        // nb_nodes:6 outside:00100011 ops:[4,5],[1,2],2,3,4
      }
      case_59: {
        // nb_nodes:6 outside:00100111 ops:4,[4,5],[2,3],3
      }
      case_60: {
        // nb_nodes:6 outside:00101111 ops:[3,4],4,[4,5]
      }
      case_61: {
        // nb_nodes:6 outside:00110000 ops:0,1,2,3,[3,4],[5,0]
      }
      case_62: {
        // nb_nodes:6 outside:00110001 ops:[0,1],1,2,3,[3,4]
      }
      case_63: {
        // nb_nodes:6 outside:00110011 ops:[3,4],[1,2],2,3
      }
      case_64: {
        // nb_nodes:6 outside:00110111 ops:[2,3],3,[3,4]
      }
      case_65: {
        // nb_nodes:6 outside:00111000 ops:0,1,2,[2,3],[5,0]
      }
      case_66: {
        // nb_nodes:6 outside:00111001 ops:[0,1],1,2,[2,3]
      }
      case_67: {
        // nb_nodes:6 outside:00111011 ops:[2,3],[1,2],2
      }
      case_68: {
        // nb_nodes:6 outside:00111100 ops:0,1,[1,2],[5,0]
      }
      case_69: {
        // nb_nodes:6 outside:00111101 ops:[0,1],1,[1,2]
      }
      case_70: {
        // nb_nodes:6 outside:00111110 ops:0,[0,1],[5,0]
      }
      case_71: {
        // nb_nodes:7 outside:00000001 ops:[0,1],1,2,3,4,5,6,[6,0]
      }
      case_72: {
        // nb_nodes:7 outside:00000010 ops:[0,1],[1,2],2,3,4,5,6,0
      }
      case_73: {
        // nb_nodes:7 outside:00000011 ops:[6,0],[1,2],2,3,4,5,6
      }
      case_74: {
        // nb_nodes:7 outside:00000100 ops:1,[1,2],[2,3],3,4,5,6,0
      }
      case_75: {
        // nb_nodes:7 outside:00000110 ops:0,[0,1],[2,3],3,4,5,6
      }
      case_76: {
        // nb_nodes:7 outside:00000111 ops:6,[6,0],[2,3],3,4,5
      }
      case_77: {
        // nb_nodes:7 outside:00001000 ops:0,1,2,[2,3],[3,4],4,5,6
      }
      case_78: {
        // nb_nodes:7 outside:00001100 ops:0,1,[1,2],[3,4],4,5,6
      }
      case_79: {
        // nb_nodes:7 outside:00001110 ops:6,0,[0,1],[3,4],4,5
      }
      case_80: {
        // nb_nodes:7 outside:00001111 ops:5,6,[6,0],[3,4],4
      }
      case_81: {
        // nb_nodes:7 outside:00010000 ops:0,1,2,3,[3,4],[4,5],5,6
      }
      case_82: {
        // nb_nodes:7 outside:00011000 ops:0,1,2,[2,3],[4,5],5,6
      }
      case_83: {
        // nb_nodes:7 outside:00011100 ops:0,1,[1,2],[4,5],5,6
      }
      case_84: {
        // nb_nodes:7 outside:00011110 ops:0,[0,1],[4,5],5,6
      }
      case_85: {
        // nb_nodes:7 outside:00011111 ops:[4,5],5,6,[6,0]
      }
      case_86: {
        // nb_nodes:7 outside:00100000 ops:0,1,2,3,4,[4,5],[5,6],6
      }
      case_87: {
        // nb_nodes:7 outside:00110000 ops:0,1,2,3,[3,4],[5,6],6
      }
      case_88: {
        // nb_nodes:7 outside:00111000 ops:0,1,2,[2,3],[5,6],6
      }
      case_89: {
        // nb_nodes:7 outside:00111100 ops:0,1,[1,2],[5,6],6
      }
      case_90: {
        // nb_nodes:7 outside:00111110 ops:0,[0,1],[5,6],6
      }
      case_91: {
        // nb_nodes:7 outside:00111111 ops:[5,6],6,[6,0]
      }
      case_92: {
        // nb_nodes:7 outside:01000000 ops:0,1,2,3,4,5,[5,6],[6,0]
      }
      case_93: {
        // nb_nodes:7 outside:01000001 ops:[0,1],1,2,3,4,5,[5,6]
      }
      case_94: {
        // nb_nodes:7 outside:01000011 ops:[5,6],[1,2],2,3,4,5
      }
      case_95: {
        // nb_nodes:7 outside:01000111 ops:5,[5,6],[2,3],3,4
      }
      case_96: {
        // nb_nodes:7 outside:01001111 ops:[3,4],4,5,[5,6]
      }
      case_97: {
        // nb_nodes:7 outside:01011111 ops:[4,5],5,[5,6]
      }
      case_98: {
        // nb_nodes:7 outside:01100000 ops:0,1,2,3,4,[4,5],[6,0]
      }
      case_99: {
        // nb_nodes:7 outside:01100001 ops:[0,1],1,2,3,4,[4,5]
      }
      case_100: {
        // nb_nodes:7 outside:01100011 ops:[4,5],[1,2],2,3,4
      }
      case_101: {
        // nb_nodes:7 outside:01100111 ops:4,[4,5],[2,3],3
      }
      case_102: {
        // nb_nodes:7 outside:01101111 ops:[3,4],4,[4,5]
      }
      case_103: {
        // nb_nodes:7 outside:01110000 ops:0,1,2,3,[3,4],[6,0]
      }
      case_104: {
        // nb_nodes:7 outside:01110001 ops:[0,1],1,2,3,[3,4]
      }
      case_105: {
        // nb_nodes:7 outside:01110011 ops:[3,4],[1,2],2,3
      }
      case_106: {
        // nb_nodes:7 outside:01110111 ops:[2,3],3,[3,4]
      }
      case_107: {
        // nb_nodes:7 outside:01111000 ops:0,1,2,[2,3],[6,0]
      }
      case_108: {
        // nb_nodes:7 outside:01111001 ops:[0,1],1,2,[2,3]
      }
      case_109: {
        // nb_nodes:7 outside:01111011 ops:[2,3],[1,2],2
      }
      case_110: {
        // nb_nodes:7 outside:01111100 ops:0,1,[1,2],[6,0]
      }
      case_111: {
        // nb_nodes:7 outside:01111101 ops:[0,1],1,[1,2]
      }
      case_112: {
        // nb_nodes:7 outside:01111110 ops:0,[0,1],[6,0]
      }
      case_113: {
        // nb_nodes:8 outside:00000001 ops:[0,1],1,2,3,4,5,6,7,[7,0]
      }
      case_114: {
        // nb_nodes:8 outside:00000010 ops:[0,1],[1,2],2,3,4,5,6,7,0
      }
      case_115: {
        // nb_nodes:8 outside:00000011 ops:[7,0],[1,2],2,3,4,5,6,7
      }
      case_116: {
        // nb_nodes:8 outside:00000100 ops:1,[1,2],[2,3],3,4,5,6,7,0
      }
      case_117: {
        // nb_nodes:8 outside:00000110 ops:0,[0,1],[2,3],3,4,5,6,7
      }
      case_118: {
        // nb_nodes:8 outside:00000111 ops:7,[7,0],[2,3],3,4,5,6
      }
      case_119: {
        // nb_nodes:8 outside:00001000 ops:1,2,[2,3],[3,4],4,5,6,7,0
      }
      case_120: {
        // nb_nodes:8 outside:00001100 ops:0,1,[1,2],[3,4],4,5,6,7
      }
      case_121: {
        // nb_nodes:8 outside:00001110 ops:7,0,[0,1],[3,4],4,5,6
      }
      case_122: {
        // nb_nodes:8 outside:00001111 ops:6,7,[7,0],[3,4],4,5
      }
      case_123: {
        // nb_nodes:8 outside:00010000 ops:0,1,2,3,[3,4],[4,5],5,6,7
      }
      case_124: {
        // nb_nodes:8 outside:00011000 ops:0,1,2,[2,3],[4,5],5,6,7
      }
      case_125: {
        // nb_nodes:8 outside:00011100 ops:0,1,[1,2],[4,5],5,6,7
      }
      case_126: {
        // nb_nodes:8 outside:00011110 ops:0,[0,1],[4,5],5,6,7
      }
      case_127: {
        // nb_nodes:8 outside:00011111 ops:[4,5],5,6,7,[7,0]
      }
      case_128: {
        // nb_nodes:8 outside:00100000 ops:0,1,2,3,4,[4,5],[5,6],6,7
      }
      case_129: {
        // nb_nodes:8 outside:00110000 ops:0,1,2,3,[3,4],[5,6],6,7
      }
      case_130: {
        // nb_nodes:8 outside:00111000 ops:0,1,2,[2,3],[5,6],6,7
      }
      case_131: {
        // nb_nodes:8 outside:00111100 ops:0,1,[1,2],[5,6],6,7
      }
      case_132: {
        // nb_nodes:8 outside:00111110 ops:0,[0,1],[5,6],6,7
      }
      case_133: {
        // nb_nodes:8 outside:00111111 ops:[5,6],6,7,[7,0]
      }
      case_134: {
        // nb_nodes:8 outside:01000000 ops:0,1,2,3,4,5,[5,6],[6,7],7
      }
      case_135: {
        // nb_nodes:8 outside:01100000 ops:0,1,2,3,4,[4,5],[6,7],7
      }
      case_136: {
        // nb_nodes:8 outside:01110000 ops:0,1,2,3,[3,4],[6,7],7
      }
      case_137: {
        // nb_nodes:8 outside:01111000 ops:0,1,2,[2,3],[6,7],7
      }
      case_138: {
        // nb_nodes:8 outside:01111100 ops:0,1,[1,2],[6,7],7
      }
      case_139: {
        // nb_nodes:8 outside:01111110 ops:0,[0,1],[6,7],7
      }
      case_140: {
        // nb_nodes:8 outside:01111111 ops:[6,7],7,[7,0]
      }
      case_141: {
        // nb_nodes:8 outside:10000000 ops:0,1,2,3,4,5,6,[6,7],[7,0]
      }
      case_142: {
        // nb_nodes:8 outside:10000001 ops:[0,1],1,2,3,4,5,6,[6,7]
      }
      case_143: {
        // nb_nodes:8 outside:10000011 ops:[6,7],[1,2],2,3,4,5,6
      }
      case_144: {
        // nb_nodes:8 outside:10000111 ops:6,[6,7],[2,3],3,4,5
      }
      case_145: {
        // nb_nodes:8 outside:10001111 ops:5,6,[6,7],[3,4],4
      }
      case_146: {
        // nb_nodes:8 outside:10011111 ops:[4,5],5,6,[6,7]
      }
      case_147: {
        // nb_nodes:8 outside:10111111 ops:[5,6],6,[6,7]
      }
      case_148: {
        // nb_nodes:8 outside:11000000 ops:0,1,2,3,4,5,[5,6],[7,0]
      }
      case_149: {
        // nb_nodes:8 outside:11000001 ops:[0,1],1,2,3,4,5,[5,6]
      }
      case_150: {
        // nb_nodes:8 outside:11000011 ops:[5,6],[1,2],2,3,4,5
      }
      case_151: {
        // nb_nodes:8 outside:11000111 ops:5,[5,6],[2,3],3,4
      }
      case_152: {
        // nb_nodes:8 outside:11001111 ops:[3,4],4,5,[5,6]
      }
      case_153: {
        // nb_nodes:8 outside:11011111 ops:[4,5],5,[5,6]
      }
      case_154: {
        // nb_nodes:8 outside:11100000 ops:0,1,2,3,4,[4,5],[7,0]
      }
      case_155: {
        // nb_nodes:8 outside:11100001 ops:[0,1],1,2,3,4,[4,5]
      }
      case_156: {
        // nb_nodes:8 outside:11100011 ops:[4,5],[1,2],2,3,4
      }
      case_157: {
        // nb_nodes:8 outside:11100111 ops:4,[4,5],[2,3],3
      }
      case_158: {
        // nb_nodes:8 outside:11101111 ops:[3,4],4,[4,5]
      }
      case_159: {
        // nb_nodes:8 outside:11110000 ops:0,1,2,3,[3,4],[7,0]
      }
      case_160: {
        // nb_nodes:8 outside:11110001 ops:[0,1],1,2,3,[3,4]
      }
      case_161: {
        // nb_nodes:8 outside:11110011 ops:[3,4],[1,2],2,3
      }
      case_162: {
        // nb_nodes:8 outside:11110111 ops:[2,3],3,[3,4]
      }
      case_163: {
        // nb_nodes:8 outside:11111000 ops:0,1,2,[2,3],[7,0]
      }
      case_164: {
        // nb_nodes:8 outside:11111001 ops:[0,1],1,2,[2,3]
      }
      case_165: {
        // nb_nodes:8 outside:11111011 ops:[2,3],[1,2],2
      }
      case_166: {
        // nb_nodes:8 outside:11111100 ops:0,1,[1,2],[7,0]
      }
      case_167: {
        // nb_nodes:8 outside:11111101 ops:[0,1],1,[1,2]
      }
      case_168: {
        // nb_nodes:8 outside:11111110 ops:0,[0,1],[7,0]
      }
      case_0: // handled in the next loop
        VF::store_aligned( nodes.xs + 0, px_0 );
        VF::store_aligned( nodes.ys + 0, py_0 );
        VC::store_aligned( nodes.cut_ids + 0, pc_0 );
        VF::store_aligned( nodes.xs + 4, px_1 );
        VF::store_aligned( nodes.ys + 4, py_1 );
        VC::store_aligned( nodes.cut_ids + 4, pc_1 );
        break;
    }
#ifdef LOC_PARSE
  }
};
} // namespace sdot
#endif // LOC_PARSE
