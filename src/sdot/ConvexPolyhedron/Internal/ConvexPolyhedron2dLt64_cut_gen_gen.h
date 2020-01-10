// simd_size: 1 make_di: 0 pi_in_regs:1
template<class TF,class TC>
bool ConvexPolyhedron2d_Lt9cut( TF *px, TF *py, TC *pi, int &nodes_size, const TF *cut_x, const TF *cut_y, const TF *cut_s, const TC *cut_i, int cut_n ) {
    using VF = SimdVec<TF,1>;
    using VC = SimdVec<TC,1>;

    VF px_0, px_1, px_2, px_3, px_4, px_5, px_6, px_7, px_8;
    VF py_0, py_1, py_2, py_3, py_4, py_5, py_6, py_7, py_8;
    VC pi_0, pi_1, pi_2, pi_3, pi_4, pi_5, pi_6, pi_7, pi_8;

    px_0 = VF::load_aligned( px + 0 );
    py_0 = VF::load_aligned( py + 0 );
    pi_0 = VC::load_aligned( pi + 0 );
    px_1 = VF::load_aligned( px + 1 );
    py_1 = VF::load_aligned( py + 1 );
    pi_1 = VC::load_aligned( pi + 1 );
    px_2 = VF::load_aligned( px + 2 );
    py_2 = VF::load_aligned( py + 2 );
    pi_2 = VC::load_aligned( pi + 2 );
    px_3 = VF::load_aligned( px + 3 );
    py_3 = VF::load_aligned( py + 3 );
    pi_3 = VC::load_aligned( pi + 3 );
    px_4 = VF::load_aligned( px + 4 );
    py_4 = VF::load_aligned( py + 4 );
    pi_4 = VC::load_aligned( pi + 4 );
    px_5 = VF::load_aligned( px + 5 );
    py_5 = VF::load_aligned( py + 5 );
    pi_5 = VC::load_aligned( pi + 5 );
    if ( nodes_size >= 6 ) {
        px_6 = VF::load_aligned( px + 6 );
        py_6 = VF::load_aligned( py + 6 );
        pi_6 = VC::load_aligned( pi + 6 );
        if ( nodes_size >= 7 ) {
            px_7 = VF::load_aligned( px + 7 );
            py_7 = VF::load_aligned( py + 7 );
            pi_7 = VC::load_aligned( pi + 7 );
            if ( nodes_size >= 8 ) {
                px_8 = VF::load_aligned( px + 8 );
                py_8 = VF::load_aligned( py + 8 );
                pi_8 = VC::load_aligned( pi + 8 );
            }
        }
    }

    auto store = [&]() {
        VF::store_aligned( px + 0, px_0 );
        VF::store_aligned( py + 0, py_0 );
        VC::store_aligned( pi + 0, pi_0 );
        VF::store_aligned( px + 1, px_1 );
        VF::store_aligned( py + 1, py_1 );
        VC::store_aligned( pi + 1, pi_1 );
        VF::store_aligned( px + 2, px_2 );
        VF::store_aligned( py + 2, py_2 );
        VC::store_aligned( pi + 2, pi_2 );
        VF::store_aligned( px + 3, px_3 );
        VF::store_aligned( py + 3, py_3 );
        VC::store_aligned( pi + 3, pi_3 );
        VF::store_aligned( px + 4, px_4 );
        VF::store_aligned( py + 4, py_4 );
        VC::store_aligned( pi + 4, pi_4 );
        VF::store_aligned( px + 5, px_5 );
        VF::store_aligned( py + 5, py_5 );
        VC::store_aligned( pi + 5, pi_5 );
        if ( nodes_size >= 6 ) {
            VF::store_aligned( px + 6, px_6 );
            VF::store_aligned( py + 6, py_6 );
            VC::store_aligned( pi + 6, pi_6 );
            if ( nodes_size >= 7 ) {
                VF::store_aligned( px + 7, px_7 );
                VF::store_aligned( py + 7, py_7 );
                VC::store_aligned( pi + 7, pi_7 );
                if ( nodes_size >= 8 ) {
                    VF::store_aligned( px + 8, px_8 );
                    VF::store_aligned( py + 8, py_8 );
                    VC::store_aligned( pi + 8, pi_8 );
                }
            }
        }
    };

    for( int num_cut = 0; ; ++num_cut ) {
        if ( num_cut == cut_n ) {
            store();
            return true;
        }

        VC ci = cut_i[ num_cut ];
        VF cx = cut_x[ num_cut ];
        VF cy = cut_y[ num_cut ];
        VF cs = cut_s[ num_cut ];
        
        int outside_nodes = 0;
        int nmsk = 1 << nodes_size;
        VF bi_0 = px_0 * cx + px_0 * cy; outside_nodes |= ( bi_0 > cs ) << 0; VF di_0 = bi_0 - cs;
        VF bi_1 = px_1 * cx + px_1 * cy; outside_nodes |= ( bi_1 > cs ) << 1; VF di_1 = bi_1 - cs;
        VF bi_2 = px_2 * cx + px_2 * cy; outside_nodes |= ( bi_2 > cs ) << 2; VF di_2 = bi_2 - cs;
        VF bi_3 = px_3 * cx + px_3 * cy; outside_nodes |= ( bi_3 > cs ) << 3; VF di_3 = bi_3 - cs;
        VF bi_4 = px_4 * cx + px_4 * cy; outside_nodes |= ( bi_4 > cs ) << 4; VF di_4 = bi_4 - cs;
        VF bi_5 = px_5 * cx + px_5 * cy; outside_nodes |= ( bi_5 > cs ) << 5; VF di_5 = bi_5 - cs;
        VF bi_6 = px_6 * cx + px_6 * cy; outside_nodes |= ( bi_6 > cs ) << 6; VF di_6 = bi_6 - cs;
        VF bi_7 = px_7 * cx + px_7 * cy; outside_nodes |= ( bi_7 > cs ) << 7; VF di_7 = bi_7 - cs;
        VF bi_8 = px_8 * cx + px_8 * cy; outside_nodes |= ( bi_8 > cs ) << 8; VF di_8 = bi_8 - cs;
        
        // if nothing has changed => go to the next cut
        int case_code = ( outside_nodes & ( nmsk - 1 ) ) | nmsk;
        if ( outside_nodes == 0 )
            continue;
        
        static void *dispatch_table[] = {
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_1   , &&case_2   , &&case_3   , &&case_4   , &&case_5   , &&case_6   , &&case_7   , &&case_8   , 
            &&case_1   , &&case_9   , &&case_10  , &&case_11  , &&case_12  , &&case_nhdl, &&case_13  , &&case_14  , 
            &&case_15  , &&case_16  , &&case_nhdl, &&case_17  , &&case_18  , &&case_19  , &&case_20  , &&case_8   , 
            &&case_1   , &&case_21  , &&case_22  , &&case_23  , &&case_24  , &&case_nhdl, &&case_25  , &&case_26  , 
            &&case_27  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_28  , &&case_nhdl, &&case_29  , &&case_30  , 
            &&case_31  , &&case_32  , &&case_nhdl, &&case_33  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_34  , 
            &&case_35  , &&case_36  , &&case_nhdl, &&case_37  , &&case_38  , &&case_39  , &&case_40  , &&case_8   , 
            &&case_1   , &&case_41  , &&case_42  , &&case_43  , &&case_44  , &&case_nhdl, &&case_45  , &&case_46  , 
            &&case_47  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_48  , &&case_nhdl, &&case_49  , &&case_50  , 
            &&case_51  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_52  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_53  , &&case_nhdl, &&case_54  , &&case_55  , 
            &&case_56  , &&case_57  , &&case_nhdl, &&case_58  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_59  , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_60  , 
            &&case_61  , &&case_62  , &&case_nhdl, &&case_63  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_64  , 
            &&case_65  , &&case_66  , &&case_nhdl, &&case_67  , &&case_68  , &&case_69  , &&case_70  , &&case_8   , 
            &&case_1   , &&case_71  , &&case_72  , &&case_73  , &&case_74  , &&case_nhdl, &&case_75  , &&case_76  , 
            &&case_77  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_78  , &&case_nhdl, &&case_79  , &&case_80  , 
            &&case_81  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_82  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_83  , &&case_nhdl, &&case_84  , &&case_85  , 
            &&case_86  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_87  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_88  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_89  , &&case_nhdl, &&case_90  , &&case_91  , 
            &&case_92  , &&case_93  , &&case_nhdl, &&case_94  , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_95  , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_96  , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_97  , 
            &&case_98  , &&case_99  , &&case_nhdl, &&case_100 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_101 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_102 , 
            &&case_103 , &&case_104 , &&case_nhdl, &&case_105 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_106 , 
            &&case_107 , &&case_108 , &&case_nhdl, &&case_109 , &&case_110 , &&case_111 , &&case_112 , &&case_8   , 
            &&case_1   , &&case_113 , &&case_114 , &&case_115 , &&case_116 , &&case_nhdl, &&case_117 , &&case_118 , 
            &&case_119 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_120 , &&case_nhdl, &&case_121 , &&case_122 , 
            &&case_123 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_124 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_125 , &&case_nhdl, &&case_126 , &&case_127 , 
            &&case_128 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_129 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_130 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_131 , &&case_nhdl, &&case_132 , &&case_133 , 
            &&case_134 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_135 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_136 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_137 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_138 , &&case_nhdl, &&case_139 , &&case_140 , 
            &&case_141 , &&case_142 , &&case_nhdl, &&case_143 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_144 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_145 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_146 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_147 , 
            &&case_148 , &&case_149 , &&case_nhdl, &&case_150 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_151 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_152 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_153 , 
            &&case_154 , &&case_155 , &&case_nhdl, &&case_156 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_157 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_158 , 
            &&case_159 , &&case_160 , &&case_nhdl, &&case_161 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_162 , 
            &&case_163 , &&case_164 , &&case_nhdl, &&case_165 , &&case_166 , &&case_167 , &&case_168 , &&case_8   , 
            &&case_1   , &&case_169 , &&case_170 , &&case_171 , &&case_172 , &&case_nhdl, &&case_173 , &&case_174 , 
            &&case_175 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_176 , &&case_nhdl, &&case_177 , &&case_178 , 
            &&case_179 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_180 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_181 , &&case_nhdl, &&case_182 , &&case_183 , 
            &&case_184 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_185 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_186 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_187 , &&case_nhdl, &&case_188 , &&case_189 , 
            &&case_190 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_191 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_192 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_193 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_194 , &&case_nhdl, &&case_195 , &&case_196 , 
            &&case_197 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_198 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_199 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_200 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_201 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_202 , &&case_nhdl, &&case_203 , &&case_204 , 
            &&case_205 , &&case_206 , &&case_nhdl, &&case_207 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_208 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_209 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_210 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_211 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_212 , 
            &&case_213 , &&case_214 , &&case_nhdl, &&case_215 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_216 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_217 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_218 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_219 , 
            &&case_220 , &&case_221 , &&case_nhdl, &&case_222 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_223 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_224 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_225 , 
            &&case_226 , &&case_227 , &&case_nhdl, &&case_228 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_229 , 
            &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_230 , 
            &&case_231 , &&case_232 , &&case_nhdl, &&case_233 , &&case_nhdl, &&case_nhdl, &&case_nhdl, &&case_234 , 
            &&case_235 , &&case_236 , &&case_nhdl, &&case_237 , &&case_238 , &&case_239 , &&case_240 , &&case_8   , 
        };
        
        goto *dispatch_table[ case_code ];
        
        case_nhdl: {
            store();
            return false;
        }
        
        case_1: {
            // fully inside (should not happen at this point)
            continue;
        }
        
        case_8: {
            // fully outside
            nodes_size = 0;
            return true;
        }
        
        case_205: {
            // out: 0,0,0,0,0,0,0,0,1 cut: 1,2,3,4,5,6,7,[7,8],[8,0],0
            nodes_size = 10;
            TF d_8_7 = di_8[ 0 ] / ( di_7[ 0 ] - di_8[ 0 ] );
            TF d_0_8 = di_0[ 0 ] / ( di_8[ 0 ] - di_0[ 0 ] );
            px[ 9 ] = px_0[ 0 ];
            py[ 9 ] = py_0[ 0 ];
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            px_1[ 0 ] = px_2[ 0 ];
            py_1[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_7[ 0 ];
            py_6[ 0 ] = py_7[ 0 ];
            px_7[ 0 ] = px_8[ 0 ] + d_8_7 * ( px_8[ 0 ] - px_7[ 0 ] );
            py_7[ 0 ] = py_8[ 0 ] + d_8_7 * ( py_8[ 0 ] - py_7[ 0 ] );
            px_8[ 0 ] = px_0[ 0 ] + d_0_8 * ( px_0[ 0 ] - px_8[ 0 ] );
            py_8[ 0 ] = py_0[ 0 ] + d_0_8 * ( py_0[ 0 ] - py_8[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_8[ 0 ] = cut_i[ num_cut ];
            pi[ 9 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_7[ 0 ];
            pi_7[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_141: {
            // out: 0,0,0,0,0,0,0,1 cut: 0,1,2,3,4,5,6,[6,7],[7,0]
            nodes_size = 9;
            TF d_7_6 = di_7[ 0 ] / ( di_6[ 0 ] - di_7[ 0 ] );
            TF d_7_0 = di_7[ 0 ] / ( di_0[ 0 ] - di_7[ 0 ] );
            px_8[ 0 ] = px_7[ 0 ] + d_7_0 * ( px_7[ 0 ] - px_0[ 0 ] );
            py_8[ 0 ] = py_7[ 0 ] + d_7_0 * ( py_7[ 0 ] - py_0[ 0 ] );
            px_7[ 0 ] = px_7[ 0 ] + d_7_6 * ( px_7[ 0 ] - px_6[ 0 ] );
            py_7[ 0 ] = py_7[ 0 ] + d_7_6 * ( py_7[ 0 ] - py_6[ 0 ] );
            pi_7[ 0 ] = cut_i[ num_cut ];
            pi_8[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_197: {
            // out: 0,0,0,0,0,0,0,1,0 cut: 1,2,3,4,5,6,[6,7],[7,8],8,0
            nodes_size = 10;
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            TF d_7_8 = di_7[ 0 ] / ( di_8[ 0 ] - di_7[ 0 ] );
            px[ 9 ] = px_0[ 0 ];
            py[ 9 ] = py_0[ 0 ];
            px_0[ 0 ] = px_1[ 0 ];
            py_0[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_2[ 0 ];
            py_1[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_6[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            px_7[ 0 ] = px_7[ 0 ] + d_7_8 * ( px_7[ 0 ] - px_8[ 0 ] );
            py_7[ 0 ] = py_7[ 0 ] + d_7_8 * ( py_7[ 0 ] - py_8[ 0 ] );
            pi_7[ 0 ] = cut_i[ num_cut ];
            pi[ 9 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_213: {
            // out: 0,0,0,0,0,0,0,1,1 cut: 0,1,2,3,4,5,6,[6,7],[8,0]
            TF d_7_6 = di_7[ 0 ] / ( di_6[ 0 ] - di_7[ 0 ] );
            TF d_8_0 = di_8[ 0 ] / ( di_0[ 0 ] - di_8[ 0 ] );
            px_7[ 0 ] = px_7[ 0 ] + d_7_6 * ( px_7[ 0 ] - px_6[ 0 ] );
            py_7[ 0 ] = py_7[ 0 ] + d_7_6 * ( py_7[ 0 ] - py_6[ 0 ] );
            px_8[ 0 ] = px_8[ 0 ] + d_8_0 * ( px_8[ 0 ] - px_0[ 0 ] );
            py_8[ 0 ] = py_8[ 0 ] + d_8_0 * ( py_8[ 0 ] - py_0[ 0 ] );
            pi_7[ 0 ] = cut_i[ num_cut ];
            pi_8[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_92: {
            // out: 0,0,0,0,0,0,1 cut: 0,1,2,3,4,5,[5,6],[6,0]
            nodes_size = 8;
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            TF d_6_0 = di_6[ 0 ] / ( di_0[ 0 ] - di_6[ 0 ] );
            px_7[ 0 ] = px_6[ 0 ] + d_6_0 * ( px_6[ 0 ] - px_0[ 0 ] );
            py_7[ 0 ] = py_6[ 0 ] + d_6_0 * ( py_6[ 0 ] - py_0[ 0 ] );
            px_6[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_6[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            pi_6[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_134: {
            // out: 0,0,0,0,0,0,1,0 cut: 3,4,5,[5,6],[6,7],7,0,1,2
            nodes_size = 9;
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            px_8[ 0 ] = px_2[ 0 ];
            py_8[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_5[ 0 ];
            py_2[ 0 ] = py_5[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            px_3[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_3[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            px_5[ 0 ] = px_7[ 0 ];
            py_5[ 0 ] = py_7[ 0 ];
            TF tmpx_1 = px_4[ 0 ];
            TF tmpy_1 = py_4[ 0 ];
            px_4[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_4[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            px_6[ 0 ] = px_0[ 0 ];
            py_6[ 0 ] = py_0[ 0 ];
            px_7[ 0 ] = px_1[ 0 ];
            py_7[ 0 ] = py_1[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_6[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_8[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_7[ 0 ];
            pi_7[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_190: {
            // out: 0,0,0,0,0,0,1,0,0 cut: 1,2,3,4,5,[5,6],[6,7],7,8,0
            nodes_size = 10;
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            px[ 9 ] = px_0[ 0 ];
            py[ 9 ] = py_0[ 0 ];
            px_0[ 0 ] = px_1[ 0 ];
            py_0[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_2[ 0 ];
            py_1[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_5[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_6[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_6[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            pi_6[ 0 ] = cut_i[ num_cut ];
            pi[ 9 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_148: {
            // out: 0,0,0,0,0,0,1,1 cut: 0,1,2,3,4,5,[5,6],[7,0]
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            TF d_7_0 = di_7[ 0 ] / ( di_0[ 0 ] - di_7[ 0 ] );
            px_6[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_6[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            px_7[ 0 ] = px_7[ 0 ] + d_7_0 * ( px_7[ 0 ] - px_0[ 0 ] );
            py_7[ 0 ] = py_7[ 0 ] + d_7_0 * ( py_7[ 0 ] - py_0[ 0 ] );
            pi_6[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_198: {
            // out: 0,0,0,0,0,0,1,1,0 cut: 0,1,2,3,4,5,[5,6],[7,8],8
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            TF d_7_8 = di_7[ 0 ] / ( di_8[ 0 ] - di_7[ 0 ] );
            px_6[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_6[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            px_7[ 0 ] = px_7[ 0 ] + d_7_8 * ( px_7[ 0 ] - px_8[ 0 ] );
            py_7[ 0 ] = py_7[ 0 ] + d_7_8 * ( py_7[ 0 ] - py_8[ 0 ] );
            pi_6[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_220: {
            // out: 0,0,0,0,0,0,1,1,1 cut: 0,1,2,3,4,5,[5,6],[8,0]
            nodes_size = 8;
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            TF d_8_0 = di_8[ 0 ] / ( di_0[ 0 ] - di_8[ 0 ] );
            px_6[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_6[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            px_7[ 0 ] = px_8[ 0 ] + d_8_0 * ( px_8[ 0 ] - px_0[ 0 ] );
            py_7[ 0 ] = py_8[ 0 ] + d_8_0 * ( py_8[ 0 ] - py_0[ 0 ] );
            pi_6[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_56: {
            // out: 0,0,0,0,0,1 cut: 0,1,2,3,4,[4,5],[5,0]
            nodes_size = 7;
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            TF d_0_5 = di_0[ 0 ] / ( di_5[ 0 ] - di_0[ 0 ] );
            px_6[ 0 ] = px_0[ 0 ] + d_0_5 * ( px_0[ 0 ] - px_5[ 0 ] );
            py_6[ 0 ] = py_0[ 0 ] + d_0_5 * ( py_0[ 0 ] - py_5[ 0 ] );
            px_5[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_5[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_86: {
            // out: 0,0,0,0,0,1,0 cut: 0,1,2,3,4,[4,5],[5,6],6
            nodes_size = 8;
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            px_7[ 0 ] = px_6[ 0 ];
            py_7[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_6[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            px_5[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_5[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_128: {
            // out: 0,0,0,0,0,1,0,0 cut: 2,3,4,[4,5],[5,6],6,7,0,1
            nodes_size = 9;
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_8[ 0 ] = px_1[ 0 ];
            py_8[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_3[ 0 ];
            py_1[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            px_2[ 0 ] = px_4[ 0 ];
            py_2[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_4[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_7[ 0 ];
            py_6[ 0 ] = py_7[ 0 ];
            px_7[ 0 ] = px_0[ 0 ];
            py_7[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_5[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_7[ 0 ];
            pi_7[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_8[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_184: {
            // out: 0,0,0,0,0,1,0,0,0 cut: 1,2,3,4,[4,5],[5,6],6,7,8,0
            nodes_size = 10;
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px[ 9 ] = px_0[ 0 ];
            py[ 9 ] = py_0[ 0 ];
            px_0[ 0 ] = px_1[ 0 ];
            py_0[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_2[ 0 ];
            py_1[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_5[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_5[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi[ 9 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_98: {
            // out: 0,0,0,0,0,1,1 cut: 1,2,3,4,[4,5],[6,0],0
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            TF d_6_0 = di_6[ 0 ] / ( di_0[ 0 ] - di_6[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            px_1[ 0 ] = px_2[ 0 ];
            py_1[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_5[ 0 ] = px_6[ 0 ] + d_6_0 * ( px_6[ 0 ] - px_0[ 0 ] );
            py_5[ 0 ] = py_6[ 0 ] + d_6_0 * ( py_6[ 0 ] - py_0[ 0 ] );
            px_6[ 0 ] = px_0[ 0 ];
            py_6[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_135: {
            // out: 0,0,0,0,0,1,1,0 cut: 0,1,2,3,4,[4,5],[6,7],7
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            px_5[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_5[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_6[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_6[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_191: {
            // out: 0,0,0,0,0,1,1,0,0 cut: 0,1,2,3,4,[4,5],[6,7],7,8
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            px_5[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_5[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            px_6[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_6[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_154: {
            // out: 0,0,0,0,0,1,1,1 cut: 0,1,2,3,4,[4,5],[7,0]
            nodes_size = 7;
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            TF d_0_7 = di_0[ 0 ] / ( di_7[ 0 ] - di_0[ 0 ] );
            px_5[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_5[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            px_6[ 0 ] = px_0[ 0 ] + d_0_7 * ( px_0[ 0 ] - px_7[ 0 ] );
            py_6[ 0 ] = py_0[ 0 ] + d_0_7 * ( py_0[ 0 ] - py_7[ 0 ] );
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_199: {
            // out: 0,0,0,0,0,1,1,1,0 cut: 0,1,2,3,4,[4,5],[7,8],8
            nodes_size = 8;
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            TF d_7_8 = di_7[ 0 ] / ( di_8[ 0 ] - di_7[ 0 ] );
            px_5[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_5[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_6[ 0 ] = px_7[ 0 ] + d_7_8 * ( px_7[ 0 ] - px_8[ 0 ] );
            py_6[ 0 ] = py_7[ 0 ] + d_7_8 * ( py_7[ 0 ] - py_8[ 0 ] );
            px_7[ 0 ] = px_8[ 0 ];
            py_7[ 0 ] = py_8[ 0 ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = pi_8[ 0 ];
            continue;
        }
        
        case_226: {
            // out: 0,0,0,0,0,1,1,1,1 cut: 3,4,[4,5],[8,0],0,1,2
            nodes_size = 7;
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            TF d_8_0 = di_8[ 0 ] / ( di_0[ 0 ] - di_8[ 0 ] );
            px_6[ 0 ] = px_2[ 0 ];
            py_6[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_5[ 0 ] = px_1[ 0 ];
            py_5[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_4[ 0 ];
            py_1[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_0[ 0 ];
            py_4[ 0 ] = py_0[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            px_3[ 0 ] = px_8[ 0 ] + d_8_0 * ( px_8[ 0 ] - px_0[ 0 ] );
            py_3[ 0 ] = py_8[ 0 ] + d_8_0 * ( py_8[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_5[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_31: {
            // out: 0,0,0,0,1 cut: 0,1,2,3,[3,4],[4,0]
            nodes_size = 6;
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF d_0_4 = di_0[ 0 ] / ( di_4[ 0 ] - di_0[ 0 ] );
            px_5[ 0 ] = px_0[ 0 ] + d_0_4 * ( px_0[ 0 ] - px_4[ 0 ] );
            py_5[ 0 ] = py_0[ 0 ] + d_0_4 * ( py_0[ 0 ] - py_4[ 0 ] );
            px_4[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_4[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_51: {
            // out: 0,0,0,0,1,0 cut: 0,1,2,3,[3,4],[4,5],5
            nodes_size = 7;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            px_6[ 0 ] = px_5[ 0 ];
            py_6[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_5[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            px_4[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_81: {
            // out: 0,0,0,0,1,0,0 cut: [3,4],[4,5],5,6,0,1,2,3
            nodes_size = 8;
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_7[ 0 ] = px_3[ 0 ];
            py_7[ 0 ] = py_3[ 0 ];
            TF tmpx_0 = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            TF tmpy_0 = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_6[ 0 ];
            py_3[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_2[ 0 ];
            py_6[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_5[ 0 ];
            py_2[ 0 ] = py_5[ 0 ];
            TF tmpx_1 = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            TF tmpy_1 = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_4[ 0 ] = px_0[ 0 ];
            py_4[ 0 ] = py_0[ 0 ];
            px_5[ 0 ] = px_1[ 0 ];
            py_5[ 0 ] = py_1[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_4[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_123: {
            // out: 0,0,0,0,1,0,0,0 cut: 0,1,2,3,[3,4],[4,5],5,6,7
            nodes_size = 9;
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            px_8[ 0 ] = px_7[ 0 ];
            py_8[ 0 ] = py_7[ 0 ];
            px_7[ 0 ] = px_6[ 0 ];
            py_7[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_5[ 0 ];
            py_6[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_5[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            px_4[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_4[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_8[ 0 ] = pi_7[ 0 ];
            pi_7[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_179: {
            // out: 0,0,0,0,1,0,0,0,0 cut: 1,2,3,[3,4],[4,5],5,6,7,8,0
            nodes_size = 10;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            px[ 9 ] = px_0[ 0 ];
            py[ 9 ] = py_0[ 0 ];
            px_0[ 0 ] = px_1[ 0 ];
            py_0[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_2[ 0 ];
            py_1[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_4[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_4[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi[ 9 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_61: {
            // out: 0,0,0,0,1,1 cut: 0,1,2,3,[3,4],[5,0]
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_0_5 = di_0[ 0 ] / ( di_5[ 0 ] - di_0[ 0 ] );
            px_4[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_5[ 0 ] = px_0[ 0 ] + d_0_5 * ( px_0[ 0 ] - px_5[ 0 ] );
            py_5[ 0 ] = py_0[ 0 ] + d_0_5 * ( py_0[ 0 ] - py_5[ 0 ] );
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_87: {
            // out: 0,0,0,0,1,1,0 cut: 0,1,2,3,[3,4],[5,6],6
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_4[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_4[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_5[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_5[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_129: {
            // out: 0,0,0,0,1,1,0,0 cut: 0,1,2,3,[3,4],[5,6],6,7
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            px_4[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_5[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_5[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_185: {
            // out: 0,0,0,0,1,1,0,0,0 cut: 0,1,2,3,[3,4],[5,6],6,7,8
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_4[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_5[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_5[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_103: {
            // out: 0,0,0,0,1,1,1 cut: 2,3,[3,4],[6,0],0,1
            nodes_size = 6;
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF d_6_0 = di_6[ 0 ] / ( di_0[ 0 ] - di_6[ 0 ] );
            px_5[ 0 ] = px_1[ 0 ];
            py_5[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_3[ 0 ];
            py_1[ 0 ] = py_3[ 0 ];
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_6[ 0 ] + d_6_0 * ( px_6[ 0 ] - px_0[ 0 ] );
            py_3[ 0 ] = py_6[ 0 ] + d_6_0 * ( py_6[ 0 ] - py_0[ 0 ] );
            px_4[ 0 ] = px_0[ 0 ];
            py_4[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_4[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_136: {
            // out: 0,0,0,0,1,1,1,0 cut: 3,[3,4],[6,7],7,0,1,2
            nodes_size = 7;
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF d_7_6 = di_7[ 0 ] / ( di_6[ 0 ] - di_7[ 0 ] );
            px_5[ 0 ] = px_1[ 0 ];
            py_5[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_1[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_4[ 0 ] = px_0[ 0 ];
            py_4[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = px_3[ 0 ];
            py_0[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_7[ 0 ];
            py_3[ 0 ] = py_7[ 0 ];
            TF tmpx_0 = px_7[ 0 ] + d_7_6 * ( px_7[ 0 ] - px_6[ 0 ] );
            TF tmpy_0 = py_7[ 0 ] + d_7_6 * ( py_7[ 0 ] - py_6[ 0 ] );
            px_6[ 0 ] = px_2[ 0 ];
            py_6[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = tmpx_0;
            py_2[ 0 ] = tmpy_0;
            pi_4[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_7[ 0 ];
            pi_5[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_192: {
            // out: 0,0,0,0,1,1,1,0,0 cut: 0,1,2,3,[3,4],[6,7],7,8
            nodes_size = 8;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            px_4[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_5[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_5[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            px_6[ 0 ] = px_7[ 0 ];
            py_6[ 0 ] = py_7[ 0 ];
            px_7[ 0 ] = px_8[ 0 ];
            py_7[ 0 ] = py_8[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = pi_7[ 0 ];
            pi_7[ 0 ] = pi_8[ 0 ];
            continue;
        }
        
        case_159: {
            // out: 0,0,0,0,1,1,1,1 cut: 0,1,2,3,[3,4],[7,0]
            nodes_size = 6;
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF d_7_0 = di_7[ 0 ] / ( di_0[ 0 ] - di_7[ 0 ] );
            px_4[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_4[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_5[ 0 ] = px_7[ 0 ] + d_7_0 * ( px_7[ 0 ] - px_0[ 0 ] );
            py_5[ 0 ] = py_7[ 0 ] + d_7_0 * ( py_7[ 0 ] - py_0[ 0 ] );
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_200: {
            // out: 0,0,0,0,1,1,1,1,0 cut: 0,1,2,3,[3,4],[7,8],8
            nodes_size = 7;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_8_7 = di_8[ 0 ] / ( di_7[ 0 ] - di_8[ 0 ] );
            px_4[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_5[ 0 ] = px_8[ 0 ] + d_8_7 * ( px_8[ 0 ] - px_7[ 0 ] );
            py_5[ 0 ] = py_8[ 0 ] + d_8_7 * ( py_8[ 0 ] - py_7[ 0 ] );
            px_6[ 0 ] = px_8[ 0 ];
            py_6[ 0 ] = py_8[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = pi_8[ 0 ];
            continue;
        }
        
        case_231: {
            // out: 0,0,0,0,1,1,1,1,1 cut: 0,1,2,3,[3,4],[8,0]
            nodes_size = 6;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_0_8 = di_0[ 0 ] / ( di_8[ 0 ] - di_0[ 0 ] );
            px_4[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_5[ 0 ] = px_0[ 0 ] + d_0_8 * ( px_0[ 0 ] - px_8[ 0 ] );
            py_5[ 0 ] = py_0[ 0 ] + d_0_8 * ( py_0[ 0 ] - py_8[ 0 ] );
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_15: {
            // out: 0,0,0,1 cut: 0,1,2,[2,3],[3,0]
            nodes_size = 5;
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_0_3 = di_0[ 0 ] / ( di_3[ 0 ] - di_0[ 0 ] );
            px_4[ 0 ] = px_0[ 0 ] + d_0_3 * ( px_0[ 0 ] - px_3[ 0 ] );
            py_4[ 0 ] = py_0[ 0 ] + d_0_3 * ( py_0[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_3[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_27: {
            // out: 0,0,0,1,0 cut: 2,[2,3],[3,4],4,0,1
            nodes_size = 6;
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            px_5[ 0 ] = px_1[ 0 ];
            py_5[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_1[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_0[ 0 ];
            py_4[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_47: {
            // out: 0,0,0,1,0,0 cut: 2,[2,3],[3,4],4,5,0,1
            nodes_size = 7;
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            px_6[ 0 ] = px_1[ 0 ];
            py_6[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_1[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_0[ 0 ];
            py_5[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_77: {
            // out: 0,0,0,1,0,0,0 cut: [2,3],[3,4],4,5,6,0,1,2
            nodes_size = 8;
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            px_7[ 0 ] = px_2[ 0 ];
            py_7[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            TF tmpy_0 = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ];
            py_2[ 0 ] = py_4[ 0 ];
            TF tmpx_1 = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            TF tmpy_1 = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            px_5[ 0 ] = px_0[ 0 ];
            py_5[ 0 ] = py_0[ 0 ];
            px_6[ 0 ] = px_1[ 0 ];
            py_6[ 0 ] = py_1[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_3[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_119: {
            // out: 0,0,0,1,0,0,0,0 cut: [2,3],[3,4],4,5,6,7,0,1,2
            nodes_size = 9;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_8[ 0 ] = px_2[ 0 ];
            py_8[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            TF tmpy_0 = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ];
            py_2[ 0 ] = py_4[ 0 ];
            TF tmpx_1 = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            TF tmpy_1 = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            px_5[ 0 ] = px_7[ 0 ];
            py_5[ 0 ] = py_7[ 0 ];
            px_6[ 0 ] = px_0[ 0 ];
            py_6[ 0 ] = py_0[ 0 ];
            px_7[ 0 ] = px_1[ 0 ];
            py_7[ 0 ] = py_1[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_3[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_7[ 0 ];
            pi_7[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_8[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_175: {
            // out: 0,0,0,1,0,0,0,0,0 cut: 2,[2,3],[3,4],4,5,6,7,8,0,1
            nodes_size = 10;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            px[ 9 ] = px_1[ 0 ];
            py[ 9 ] = py_1[ 0 ];
            px_1[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_1[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_7[ 0 ];
            py_6[ 0 ] = py_7[ 0 ];
            px_7[ 0 ] = px_8[ 0 ];
            py_7[ 0 ] = py_8[ 0 ];
            px_8[ 0 ] = px_0[ 0 ];
            py_8[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_7[ 0 ];
            pi_7[ 0 ] = pi_8[ 0 ];
            pi_8[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi[ 9 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_35: {
            // out: 0,0,0,1,1 cut: [4,0],0,1,2,[2,3]
            TF d_4_0 = di_4[ 0 ] / ( di_0[ 0 ] - di_4[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF tmpx_0 = px_4[ 0 ] + d_4_0 * ( px_4[ 0 ] - px_0[ 0 ] );
            TF tmpy_0 = py_4[ 0 ] + d_4_0 * ( py_4[ 0 ] - py_0[ 0 ] );
            px_4[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_4[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_3[ 0 ] = px_2[ 0 ];
            py_3[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_1[ 0 ];
            py_2[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_0[ 0 ];
            py_1[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_3[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_52: {
            // out: 0,0,0,1,1,0 cut: 0,1,2,[2,3],[4,5],5
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_3[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_3[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_4[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_82: {
            // out: 0,0,0,1,1,0,0 cut: 0,1,2,[2,3],[4,5],5,6
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_3[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_4[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_124: {
            // out: 0,0,0,1,1,0,0,0 cut: 0,1,2,[2,3],[4,5],5,6,7
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            px_3[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_3[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_4[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_4[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_180: {
            // out: 0,0,0,1,1,0,0,0,0 cut: 0,1,2,[2,3],[4,5],5,6,7,8
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            px_3[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_4[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_4[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_65: {
            // out: 0,0,0,1,1,1 cut: 0,1,2,[2,3],[5,0]
            nodes_size = 5;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_5_0 = di_5[ 0 ] / ( di_0[ 0 ] - di_5[ 0 ] );
            px_3[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_4[ 0 ] = px_5[ 0 ] + d_5_0 * ( px_5[ 0 ] - px_0[ 0 ] );
            py_4[ 0 ] = py_5[ 0 ] + d_5_0 * ( py_5[ 0 ] - py_0[ 0 ] );
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_88: {
            // out: 0,0,0,1,1,1,0 cut: 0,1,2,[2,3],[5,6],6
            nodes_size = 6;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            px_3[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_4[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_4[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = pi_6[ 0 ];
            continue;
        }
        
        case_130: {
            // out: 0,0,0,1,1,1,0,0 cut: 0,1,2,[2,3],[5,6],6,7
            nodes_size = 7;
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_3[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_3[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_4[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_4[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_7[ 0 ];
            py_6[ 0 ] = py_7[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_7[ 0 ];
            continue;
        }
        
        case_186: {
            // out: 0,0,0,1,1,1,0,0,0 cut: 2,[2,3],[5,6],6,7,8,0,1
            nodes_size = 8;
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            px_4[ 0 ] = px_7[ 0 ];
            py_4[ 0 ] = py_7[ 0 ];
            px_7[ 0 ] = px_1[ 0 ];
            py_7[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_1[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_3[ 0 ] = px_6[ 0 ];
            py_3[ 0 ] = py_6[ 0 ];
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            px_2[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            px_5[ 0 ] = px_8[ 0 ];
            py_5[ 0 ] = py_8[ 0 ];
            px_6[ 0 ] = px_0[ 0 ];
            py_6[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_3[ 0 ] = pi_6[ 0 ];
            pi_4[ 0 ] = pi_7[ 0 ];
            pi_5[ 0 ] = pi_8[ 0 ];
            pi_6[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_107: {
            // out: 0,0,0,1,1,1,1 cut: [2,3],[6,0],0,1,2
            nodes_size = 5;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_6_0 = di_6[ 0 ] / ( di_0[ 0 ] - di_6[ 0 ] );
            px_4[ 0 ] = px_2[ 0 ];
            py_4[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            TF tmpy_0 = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            px_3[ 0 ] = px_1[ 0 ];
            py_3[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_6[ 0 ] + d_6_0 * ( px_6[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_6[ 0 ] + d_6_0 * ( py_6[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_3[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_137: {
            // out: 0,0,0,1,1,1,1,0 cut: 0,1,2,[2,3],[6,7],7
            nodes_size = 6;
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_7_6 = di_7[ 0 ] / ( di_6[ 0 ] - di_7[ 0 ] );
            px_3[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_3[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_4[ 0 ] = px_7[ 0 ] + d_7_6 * ( px_7[ 0 ] - px_6[ 0 ] );
            py_4[ 0 ] = py_7[ 0 ] + d_7_6 * ( py_7[ 0 ] - py_6[ 0 ] );
            px_5[ 0 ] = px_7[ 0 ];
            py_5[ 0 ] = py_7[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = pi_7[ 0 ];
            continue;
        }
        
        case_193: {
            // out: 0,0,0,1,1,1,1,0,0 cut: 0,1,2,[2,3],[6,7],7,8
            nodes_size = 7;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_7_6 = di_7[ 0 ] / ( di_6[ 0 ] - di_7[ 0 ] );
            px_3[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_4[ 0 ] = px_7[ 0 ] + d_7_6 * ( px_7[ 0 ] - px_6[ 0 ] );
            py_4[ 0 ] = py_7[ 0 ] + d_7_6 * ( py_7[ 0 ] - py_6[ 0 ] );
            px_5[ 0 ] = px_7[ 0 ];
            py_5[ 0 ] = py_7[ 0 ];
            px_6[ 0 ] = px_8[ 0 ];
            py_6[ 0 ] = py_8[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = pi_7[ 0 ];
            pi_6[ 0 ] = pi_8[ 0 ];
            continue;
        }
        
        case_163: {
            // out: 0,0,0,1,1,1,1,1 cut: [2,3],[7,0],0,1,2
            nodes_size = 5;
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_0_7 = di_0[ 0 ] / ( di_7[ 0 ] - di_0[ 0 ] );
            px_4[ 0 ] = px_2[ 0 ];
            py_4[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            TF tmpy_0 = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            px_3[ 0 ] = px_1[ 0 ];
            py_3[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_0[ 0 ] + d_0_7 * ( px_0[ 0 ] - px_7[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_7 * ( py_0[ 0 ] - py_7[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_3[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_201: {
            // out: 0,0,0,1,1,1,1,1,0 cut: 0,1,2,[2,3],[7,8],8
            nodes_size = 6;
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_7_8 = di_7[ 0 ] / ( di_8[ 0 ] - di_7[ 0 ] );
            px_3[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_3[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_4[ 0 ] = px_7[ 0 ] + d_7_8 * ( px_7[ 0 ] - px_8[ 0 ] );
            py_4[ 0 ] = py_7[ 0 ] + d_7_8 * ( py_7[ 0 ] - py_8[ 0 ] );
            px_5[ 0 ] = px_8[ 0 ];
            py_5[ 0 ] = py_8[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = pi_8[ 0 ];
            continue;
        }
        
        case_235: {
            // out: 0,0,0,1,1,1,1,1,1 cut: [2,3],[8,0],0,1,2
            nodes_size = 5;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_0_8 = di_0[ 0 ] / ( di_8[ 0 ] - di_0[ 0 ] );
            px_4[ 0 ] = px_2[ 0 ];
            py_4[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            TF tmpy_0 = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            px_3[ 0 ] = px_1[ 0 ];
            py_3[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_0[ 0 ] + d_0_8 * ( px_0[ 0 ] - px_8[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_8 * ( py_0[ 0 ] - py_8[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_3[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_5: {
            // out: 0,0,1 cut: 0,1,[1,2],[2,0]
            nodes_size = 4;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_0_2 = di_0[ 0 ] / ( di_2[ 0 ] - di_0[ 0 ] );
            px_3[ 0 ] = px_0[ 0 ] + d_0_2 * ( px_0[ 0 ] - px_2[ 0 ] );
            py_3[ 0 ] = py_0[ 0 ] + d_0_2 * ( py_0[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_12: {
            // out: 0,0,1,0 cut: 0,1,[1,2],[2,3],3
            nodes_size = 5;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px_4[ 0 ] = px_3[ 0 ];
            py_4[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_24: {
            // out: 0,0,1,0,0 cut: 3,4,0,1,[1,2],[2,3]
            nodes_size = 6;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_5[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_5[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            px_3[ 0 ] = px_1[ 0 ];
            py_3[ 0 ] = py_1[ 0 ];
            TF tmpx_1 = px_4[ 0 ];
            TF tmpy_1 = py_4[ 0 ];
            px_4[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_4[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_44: {
            // out: 0,0,1,0,0,0 cut: 1,[1,2],[2,3],3,4,5,0
            nodes_size = 7;
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px_6[ 0 ] = px_0[ 0 ];
            py_6[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = px_1[ 0 ];
            py_0[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_74: {
            // out: 0,0,1,0,0,0,0 cut: 5,6,0,1,[1,2],[2,3],3,4
            nodes_size = 8;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px_7[ 0 ] = px_4[ 0 ];
            py_7[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_4[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            TF tmpx_0 = px_5[ 0 ];
            TF tmpy_0 = py_5[ 0 ];
            px_5[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_5[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            TF tmpx_1 = px_6[ 0 ];
            TF tmpy_1 = py_6[ 0 ];
            px_6[ 0 ] = px_3[ 0 ];
            py_6[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_1[ 0 ];
            py_3[ 0 ] = py_1[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            TF tmpi_0 = pi_6[ 0 ];
            pi_6[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_1[ 0 ];
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            continue;
        }
        
        case_116: {
            // out: 0,0,1,0,0,0,0,0 cut: 1,[1,2],[2,3],3,4,5,6,7,0
            nodes_size = 9;
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px_8[ 0 ] = px_0[ 0 ];
            py_8[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = px_1[ 0 ];
            py_0[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_8[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_172: {
            // out: 0,0,1,0,0,0,0,0,0 cut: 1,[1,2],[2,3],3,4,5,6,7,8,0
            nodes_size = 10;
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px[ 9 ] = px_0[ 0 ];
            py[ 9 ] = py_0[ 0 ];
            px_0[ 0 ] = px_1[ 0 ];
            py_0[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi[ 9 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_18: {
            // out: 0,0,1,1 cut: [3,0],0,1,[1,2]
            TF d_0_3 = di_0[ 0 ] / ( di_3[ 0 ] - di_0[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF tmpx_0 = px_0[ 0 ] + d_0_3 * ( px_0[ 0 ] - px_3[ 0 ] );
            TF tmpy_0 = py_0[ 0 ] + d_0_3 * ( py_0[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_3[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ];
            py_2[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_0[ 0 ];
            py_1[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_2[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_28: {
            // out: 0,0,1,1,0 cut: [1,2],[3,4],4,0,1
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF tmpx_0 = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            TF tmpy_0 = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ];
            py_2[ 0 ] = py_4[ 0 ];
            TF tmpx_1 = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            TF tmpy_1 = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_0[ 0 ];
            py_3[ 0 ] = py_0[ 0 ];
            px_4[ 0 ] = px_1[ 0 ];
            py_4[ 0 ] = py_1[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_48: {
            // out: 0,0,1,1,0,0 cut: 1,[1,2],[3,4],4,5,0
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            px_1[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_1[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_0[ 0 ];
            py_5[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_78: {
            // out: 0,0,1,1,0,0,0 cut: 0,1,[1,2],[3,4],4,5,6
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_120: {
            // out: 0,0,1,1,0,0,0,0 cut: 0,1,[1,2],[3,4],4,5,6,7
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_176: {
            // out: 0,0,1,1,0,0,0,0,0 cut: 0,1,[1,2],[3,4],4,5,6,7,8
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_38: {
            // out: 0,0,1,1,1 cut: [4,0],0,1,[1,2]
            nodes_size = 4;
            TF d_4_0 = di_4[ 0 ] / ( di_0[ 0 ] - di_4[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_3[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ];
            py_2[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_0[ 0 ];
            py_1[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = px_4[ 0 ] + d_4_0 * ( px_4[ 0 ] - px_0[ 0 ] );
            py_0[ 0 ] = py_4[ 0 ] + d_4_0 * ( py_4[ 0 ] - py_0[ 0 ] );
            pi_2[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_53: {
            // out: 0,0,1,1,1,0 cut: 0,1,[1,2],[4,5],5
            nodes_size = 5;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_5[ 0 ];
            continue;
        }
        
        case_83: {
            // out: 0,0,1,1,1,0,0 cut: 0,1,[1,2],[4,5],5,6
            nodes_size = 6;
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            px_2[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_2[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_3[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_3[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_6[ 0 ];
            continue;
        }
        
        case_125: {
            // out: 0,0,1,1,1,0,0,0 cut: 0,1,[1,2],[4,5],5,6,7
            nodes_size = 7;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_3[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_3[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_7[ 0 ];
            py_6[ 0 ] = py_7[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_7[ 0 ];
            continue;
        }
        
        case_181: {
            // out: 0,0,1,1,1,0,0,0,0 cut: 8,0,1,[1,2],[4,5],5,6,7
            nodes_size = 8;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_3[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_3[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ];
            py_2[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_0[ 0 ];
            py_1[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = px_8[ 0 ];
            py_0[ 0 ] = py_8[ 0 ];
            px_4[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            pi_2[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_8[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_68: {
            // out: 0,0,1,1,1,1 cut: 0,1,[1,2],[5,0]
            nodes_size = 4;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_5_0 = di_5[ 0 ] / ( di_0[ 0 ] - di_5[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_3[ 0 ] = px_5[ 0 ] + d_5_0 * ( px_5[ 0 ] - px_0[ 0 ] );
            py_3[ 0 ] = py_5[ 0 ] + d_5_0 * ( py_5[ 0 ] - py_0[ 0 ] );
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_89: {
            // out: 0,0,1,1,1,1,0 cut: 0,1,[1,2],[5,6],6
            nodes_size = 5;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_3[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_3[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_6[ 0 ];
            continue;
        }
        
        case_131: {
            // out: 0,0,1,1,1,1,0,0 cut: 0,1,[1,2],[5,6],6,7
            nodes_size = 6;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_3[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_3[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            px_5[ 0 ] = px_7[ 0 ];
            py_5[ 0 ] = py_7[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_6[ 0 ];
            pi_5[ 0 ] = pi_7[ 0 ];
            continue;
        }
        
        case_187: {
            // out: 0,0,1,1,1,1,0,0,0 cut: 0,1,[1,2],[5,6],6,7,8
            nodes_size = 7;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_3[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_3[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            px_5[ 0 ] = px_7[ 0 ];
            py_5[ 0 ] = py_7[ 0 ];
            px_6[ 0 ] = px_8[ 0 ];
            py_6[ 0 ] = py_8[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_6[ 0 ];
            pi_5[ 0 ] = pi_7[ 0 ];
            pi_6[ 0 ] = pi_8[ 0 ];
            continue;
        }
        
        case_110: {
            // out: 0,0,1,1,1,1,1 cut: [6,0],0,1,[1,2]
            nodes_size = 4;
            TF d_0_6 = di_0[ 0 ] / ( di_6[ 0 ] - di_0[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_3[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_3[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ];
            py_2[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_0[ 0 ];
            py_1[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = px_0[ 0 ] + d_0_6 * ( px_0[ 0 ] - px_6[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_6 * ( py_0[ 0 ] - py_6[ 0 ] );
            pi_2[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_138: {
            // out: 0,0,1,1,1,1,1,0 cut: [1,2],[6,7],7,0,1
            nodes_size = 5;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            px_3[ 0 ] = px_0[ 0 ];
            py_3[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_0[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_7[ 0 ];
            py_2[ 0 ] = py_7[ 0 ];
            px_4[ 0 ] = px_1[ 0 ];
            py_4[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_1[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            pi_2[ 0 ] = pi_7[ 0 ];
            pi_3[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_194: {
            // out: 0,0,1,1,1,1,1,0,0 cut: 0,1,[1,2],[6,7],7,8
            nodes_size = 6;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_7_6 = di_7[ 0 ] / ( di_6[ 0 ] - di_7[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_3[ 0 ] = px_7[ 0 ] + d_7_6 * ( px_7[ 0 ] - px_6[ 0 ] );
            py_3[ 0 ] = py_7[ 0 ] + d_7_6 * ( py_7[ 0 ] - py_6[ 0 ] );
            px_4[ 0 ] = px_7[ 0 ];
            py_4[ 0 ] = py_7[ 0 ];
            px_5[ 0 ] = px_8[ 0 ];
            py_5[ 0 ] = py_8[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_7[ 0 ];
            pi_5[ 0 ] = pi_8[ 0 ];
            continue;
        }
        
        case_166: {
            // out: 0,0,1,1,1,1,1,1 cut: [1,2],[7,0],0,1
            nodes_size = 4;
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF d_7_0 = di_7[ 0 ] / ( di_0[ 0 ] - di_7[ 0 ] );
            px_3[ 0 ] = px_1[ 0 ];
            py_3[ 0 ] = py_1[ 0 ];
            TF tmpx_0 = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            TF tmpy_0 = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = px_7[ 0 ] + d_7_0 * ( px_7[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_7[ 0 ] + d_7_0 * ( py_7[ 0 ] - py_0[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_202: {
            // out: 0,0,1,1,1,1,1,1,0 cut: [1,2],[7,8],8,0,1
            nodes_size = 5;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_7_8 = di_7[ 0 ] / ( di_8[ 0 ] - di_7[ 0 ] );
            px_3[ 0 ] = px_0[ 0 ];
            py_3[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_0[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_8[ 0 ];
            py_2[ 0 ] = py_8[ 0 ];
            px_4[ 0 ] = px_1[ 0 ];
            py_4[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_7[ 0 ] + d_7_8 * ( px_7[ 0 ] - px_8[ 0 ] );
            py_1[ 0 ] = py_7[ 0 ] + d_7_8 * ( py_7[ 0 ] - py_8[ 0 ] );
            pi_2[ 0 ] = pi_8[ 0 ];
            pi_3[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_238: {
            // out: 0,0,1,1,1,1,1,1,1 cut: [1,2],[8,0],0,1
            nodes_size = 4;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_0_8 = di_0[ 0 ] / ( di_8[ 0 ] - di_0[ 0 ] );
            px_3[ 0 ] = px_1[ 0 ];
            py_3[ 0 ] = py_1[ 0 ];
            TF tmpx_0 = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            TF tmpy_0 = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_8 * ( px_0[ 0 ] - px_8[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_8 * ( py_0[ 0 ] - py_8[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_3: {
            // out: 0,1,0 cut: [0,1],[1,2],2,0
            nodes_size = 4;
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_3[ 0 ] = px_0[ 0 ];
            py_3[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_0[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_1[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_10: {
            // out: 0,1,0,0 cut: 2,3,0,[0,1],[1,2]
            nodes_size = 5;
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_4[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_4[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            TF tmpx_1 = px_3[ 0 ];
            TF tmpy_1 = py_3[ 0 ];
            px_3[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_3[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            TF tmpi_0 = pi_2[ 0 ];
            pi_2[ 0 ] = pi_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            continue;
        }
        
        case_22: {
            // out: 0,1,0,0,0 cut: 4,0,[0,1],[1,2],2,3
            nodes_size = 6;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_5[ 0 ] = px_3[ 0 ];
            py_5[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            TF tmpx_0 = px_4[ 0 ];
            TF tmpy_0 = py_4[ 0 ];
            px_4[ 0 ] = px_2[ 0 ];
            py_4[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ];
            py_1[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_1[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_42: {
            // out: 0,1,0,0,0,0 cut: 4,5,0,[0,1],[1,2],2,3
            nodes_size = 7;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_6[ 0 ] = px_3[ 0 ];
            py_6[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_3[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            TF tmpx_0 = px_4[ 0 ];
            TF tmpy_0 = py_4[ 0 ];
            px_4[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_4[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = px_5[ 0 ];
            py_1[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_2[ 0 ];
            py_5[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_1[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_72: {
            // out: 0,1,0,0,0,0,0 cut: [1,2],2,3,4,5,6,0,[0,1]
            nodes_size = 8;
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            px_7[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_7[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            TF tmpx_0 = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            TF tmpy_0 = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = px_2[ 0 ];
            py_1[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_0[ 0 ];
            py_6[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_114: {
            // out: 0,1,0,0,0,0,0,0 cut: [0,1],[1,2],2,3,4,5,6,7,0
            nodes_size = 9;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_8[ 0 ] = px_0[ 0 ];
            py_8[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_8[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_170: {
            // out: 0,1,0,0,0,0,0,0,0 cut: 4,5,6,7,8,0,[0,1],[1,2],2,3
            nodes_size = 10;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px[ 9 ] = px_3[ 0 ];
            py[ 9 ] = py_3[ 0 ];
            px_3[ 0 ] = px_7[ 0 ];
            py_3[ 0 ] = py_7[ 0 ];
            px_7[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_7[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            TF tmpx_0 = px_4[ 0 ];
            TF tmpy_0 = py_4[ 0 ];
            px_4[ 0 ] = px_8[ 0 ];
            py_4[ 0 ] = py_8[ 0 ];
            px_8[ 0 ] = px_2[ 0 ];
            py_8[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_6[ 0 ];
            py_2[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_6[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = px_5[ 0 ];
            py_1[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_0[ 0 ];
            py_5[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_1[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_8[ 0 ];
            pi_8[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            pi[ 9 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_7[ 0 ];
            pi_7[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_7: {
            // out: 0,1,1 cut: [2,0],0,[0,1]
            TF d_0_2 = di_0[ 0 ] / ( di_2[ 0 ] - di_0[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF tmpx_0 = px_0[ 0 ] + d_0_2 * ( px_0[ 0 ] - px_2[ 0 ] );
            TF tmpy_0 = py_0[ 0 ] + d_0_2 * ( py_0[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ];
            py_1[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_1[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_13: {
            // out: 0,1,1,0 cut: [2,3],3,0,[0,1]
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF tmpx_0 = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            TF tmpy_0 = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            TF tmpx_1 = px_3[ 0 ];
            TF tmpy_1 = py_3[ 0 ];
            px_3[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_3[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_25: {
            // out: 0,1,1,0,0 cut: 0,[0,1],[2,3],3,4
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px_1[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_2[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_45: {
            // out: 0,1,1,0,0,0 cut: 5,0,[0,1],[2,3],3,4
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF tmpx_0 = px_5[ 0 ];
            TF tmpy_0 = py_5[ 0 ];
            px_5[ 0 ] = px_4[ 0 ];
            py_5[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_3[ 0 ];
            py_4[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ];
            py_1[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_1[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_75: {
            // out: 0,1,1,0,0,0,0 cut: 0,[0,1],[2,3],3,4,5,6
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_117: {
            // out: 0,1,1,0,0,0,0,0 cut: 0,[0,1],[2,3],3,4,5,6,7
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_173: {
            // out: 0,1,1,0,0,0,0,0,0 cut: 0,[0,1],[2,3],3,4,5,6,7,8
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px_1[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_2[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_20: {
            // out: 0,1,1,1 cut: [0,1],[3,0],0
            nodes_size = 3;
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_3_0 = di_3[ 0 ] / ( di_0[ 0 ] - di_3[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            TF tmpx_0 = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            TF tmpy_0 = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_1[ 0 ] = px_3[ 0 ] + d_3_0 * ( px_3[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_3[ 0 ] + d_3_0 * ( py_3[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_29: {
            // out: 0,1,1,1,0 cut: 0,[0,1],[3,4],4
            nodes_size = 4;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            continue;
        }
        
        case_49: {
            // out: 0,1,1,1,0,0 cut: 0,[0,1],[3,4],4,5
            nodes_size = 5;
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            px_1[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            continue;
        }
        
        case_79: {
            // out: 0,1,1,1,0,0,0 cut: 5,6,0,[0,1],[3,4],4
            nodes_size = 6;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            TF tmpx_0 = px_5[ 0 ];
            TF tmpy_0 = py_5[ 0 ];
            px_5[ 0 ] = px_4[ 0 ];
            py_5[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_3[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = px_6[ 0 ];
            py_1[ 0 ] = py_6[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_1[ 0 ] = pi_6[ 0 ];
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_121: {
            // out: 0,1,1,1,0,0,0,0 cut: 0,[0,1],[3,4],4,5,6,7
            nodes_size = 7;
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_1[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_7[ 0 ];
            py_6[ 0 ] = py_7[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_7[ 0 ];
            continue;
        }
        
        case_177: {
            // out: 0,1,1,1,0,0,0,0,0 cut: 0,[0,1],[3,4],4,5,6,7,8
            nodes_size = 8;
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_1[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_7[ 0 ];
            py_6[ 0 ] = py_7[ 0 ];
            px_7[ 0 ] = px_8[ 0 ];
            py_7[ 0 ] = py_8[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_7[ 0 ];
            pi_7[ 0 ] = pi_8[ 0 ];
            continue;
        }
        
        case_40: {
            // out: 0,1,1,1,1 cut: 0,[0,1],[4,0]
            nodes_size = 3;
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_4_0 = di_4[ 0 ] / ( di_0[ 0 ] - di_4[ 0 ] );
            px_1[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_0 * ( px_4[ 0 ] - px_0[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_0 * ( py_4[ 0 ] - py_0[ 0 ] );
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_54: {
            // out: 0,1,1,1,1,0 cut: 5,0,[0,1],[4,5]
            nodes_size = 4;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ];
            py_1[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = px_5[ 0 ];
            py_0[ 0 ] = py_5[ 0 ];
            px_3[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            pi_1[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_84: {
            // out: 0,1,1,1,1,0,0 cut: 0,[0,1],[4,5],5,6
            nodes_size = 5;
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_1[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_5[ 0 ];
            pi_4[ 0 ] = pi_6[ 0 ];
            continue;
        }
        
        case_126: {
            // out: 0,1,1,1,1,0,0,0 cut: 6,7,0,[0,1],[4,5],5
            nodes_size = 6;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            px_3[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_3[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_1[ 0 ] = px_7[ 0 ];
            py_1[ 0 ] = py_7[ 0 ];
            px_4[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_4[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            pi_1[ 0 ] = pi_7[ 0 ];
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_182: {
            // out: 0,1,1,1,1,0,0,0,0 cut: 7,8,0,[0,1],[4,5],5,6
            nodes_size = 7;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            px_3[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_3[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = px_7[ 0 ];
            py_0[ 0 ] = py_7[ 0 ];
            px_1[ 0 ] = px_8[ 0 ];
            py_1[ 0 ] = py_8[ 0 ];
            px_4[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_4[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            pi_1[ 0 ] = pi_8[ 0 ];
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = pi_7[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_70: {
            // out: 0,1,1,1,1,1 cut: [5,0],0,[0,1]
            nodes_size = 3;
            TF d_5_0 = di_5[ 0 ] / ( di_0[ 0 ] - di_5[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ];
            py_1[ 0 ] = py_0[ 0 ];
            px_0[ 0 ] = px_5[ 0 ] + d_5_0 * ( px_5[ 0 ] - px_0[ 0 ] );
            py_0[ 0 ] = py_5[ 0 ] + d_5_0 * ( py_5[ 0 ] - py_0[ 0 ] );
            pi_1[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_90: {
            // out: 0,1,1,1,1,1,0 cut: 0,[0,1],[5,6],6
            nodes_size = 4;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_2[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_3[ 0 ] = px_6[ 0 ];
            py_3[ 0 ] = py_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_6[ 0 ];
            continue;
        }
        
        case_132: {
            // out: 0,1,1,1,1,1,0,0 cut: 0,[0,1],[5,6],6,7
            nodes_size = 5;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            px_3[ 0 ] = px_6[ 0 ];
            py_3[ 0 ] = py_6[ 0 ];
            px_4[ 0 ] = px_7[ 0 ];
            py_4[ 0 ] = py_7[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_6[ 0 ];
            pi_4[ 0 ] = pi_7[ 0 ];
            continue;
        }
        
        case_188: {
            // out: 0,1,1,1,1,1,0,0,0 cut: 0,[0,1],[5,6],6,7,8
            nodes_size = 6;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            px_3[ 0 ] = px_6[ 0 ];
            py_3[ 0 ] = py_6[ 0 ];
            px_4[ 0 ] = px_7[ 0 ];
            py_4[ 0 ] = py_7[ 0 ];
            px_5[ 0 ] = px_8[ 0 ];
            py_5[ 0 ] = py_8[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_6[ 0 ];
            pi_4[ 0 ] = pi_7[ 0 ];
            pi_5[ 0 ] = pi_8[ 0 ];
            continue;
        }
        
        case_112: {
            // out: 0,1,1,1,1,1,1 cut: [0,1],[6,0],0
            nodes_size = 3;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_0_6 = di_0[ 0 ] / ( di_6[ 0 ] - di_0[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ];
            py_2[ 0 ] = py_0[ 0 ];
            TF tmpx_0 = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            TF tmpy_0 = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_6 * ( px_0[ 0 ] - px_6[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_6 * ( py_0[ 0 ] - py_6[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = pi_0[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_139: {
            // out: 0,1,1,1,1,1,1,0 cut: 0,[0,1],[6,7],7
            nodes_size = 4;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_7_6 = di_7[ 0 ] / ( di_6[ 0 ] - di_7[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_7[ 0 ] + d_7_6 * ( px_7[ 0 ] - px_6[ 0 ] );
            py_2[ 0 ] = py_7[ 0 ] + d_7_6 * ( py_7[ 0 ] - py_6[ 0 ] );
            px_3[ 0 ] = px_7[ 0 ];
            py_3[ 0 ] = py_7[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_7[ 0 ];
            continue;
        }
        
        case_195: {
            // out: 0,1,1,1,1,1,1,0,0 cut: 0,[0,1],[6,7],7,8
            nodes_size = 5;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_2[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            px_3[ 0 ] = px_7[ 0 ];
            py_3[ 0 ] = py_7[ 0 ];
            px_4[ 0 ] = px_8[ 0 ];
            py_4[ 0 ] = py_8[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_7[ 0 ];
            pi_4[ 0 ] = pi_8[ 0 ];
            continue;
        }
        
        case_168: {
            // out: 0,1,1,1,1,1,1,1 cut: 0,[0,1],[7,0]
            nodes_size = 3;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_7_0 = di_7[ 0 ] / ( di_0[ 0 ] - di_7[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_7[ 0 ] + d_7_0 * ( px_7[ 0 ] - px_0[ 0 ] );
            py_2[ 0 ] = py_7[ 0 ] + d_7_0 * ( py_7[ 0 ] - py_0[ 0 ] );
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_203: {
            // out: 0,1,1,1,1,1,1,1,0 cut: 0,[0,1],[7,8],8
            nodes_size = 4;
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_7_8 = di_7[ 0 ] / ( di_8[ 0 ] - di_7[ 0 ] );
            px_1[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_2[ 0 ] = px_7[ 0 ] + d_7_8 * ( px_7[ 0 ] - px_8[ 0 ] );
            py_2[ 0 ] = py_7[ 0 ] + d_7_8 * ( py_7[ 0 ] - py_8[ 0 ] );
            px_3[ 0 ] = px_8[ 0 ];
            py_3[ 0 ] = py_8[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_8[ 0 ];
            continue;
        }
        
        case_240: {
            // out: 0,1,1,1,1,1,1,1,1 cut: 0,[0,1],[8,0]
            nodes_size = 3;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_0_8 = di_0[ 0 ] / ( di_8[ 0 ] - di_0[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ] + d_0_8 * ( px_0[ 0 ] - px_8[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_8 * ( py_0[ 0 ] - py_8[ 0 ] );
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_2: {
            // out: 1,0,0 cut: 2,[2,0],[0,1],1
            nodes_size = 4;
            TF d_0_2 = di_0[ 0 ] / ( di_2[ 0 ] - di_0[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            px_3[ 0 ] = px_1[ 0 ];
            py_3[ 0 ] = py_1[ 0 ];
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            TF tmpx_1 = px_0[ 0 ] + d_0_2 * ( px_0[ 0 ] - px_2[ 0 ] );
            TF tmpy_1 = py_0[ 0 ] + d_0_2 * ( py_0[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_9: {
            // out: 1,0,0,0 cut: [0,1],1,2,3,[3,0]
            nodes_size = 5;
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_0_3 = di_0[ 0 ] / ( di_3[ 0 ] - di_0[ 0 ] );
            px_4[ 0 ] = px_0[ 0 ] + d_0_3 * ( px_0[ 0 ] - px_3[ 0 ] );
            py_4[ 0 ] = py_0[ 0 ] + d_0_3 * ( py_0[ 0 ] - py_3[ 0 ] );
            px_0[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_0[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_21: {
            // out: 1,0,0,0,0 cut: [0,1],1,2,3,4,[4,0]
            nodes_size = 6;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_0_4 = di_0[ 0 ] / ( di_4[ 0 ] - di_0[ 0 ] );
            px_5[ 0 ] = px_0[ 0 ] + d_0_4 * ( px_0[ 0 ] - px_4[ 0 ] );
            py_5[ 0 ] = py_0[ 0 ] + d_0_4 * ( py_0[ 0 ] - py_4[ 0 ] );
            px_0[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_41: {
            // out: 1,0,0,0,0,0 cut: 1,2,3,4,5,[5,0],[0,1]
            nodes_size = 7;
            TF d_5_0 = di_5[ 0 ] / ( di_0[ 0 ] - di_5[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            px_6[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_6[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            px_1[ 0 ] = px_2[ 0 ];
            py_1[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_5[ 0 ] + d_5_0 * ( px_5[ 0 ] - px_0[ 0 ] );
            py_5[ 0 ] = py_5[ 0 ] + d_5_0 * ( py_5[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_71: {
            // out: 1,0,0,0,0,0,0 cut: 2,3,4,5,6,[6,0],[0,1],1
            nodes_size = 8;
            TF d_0_6 = di_0[ 0 ] / ( di_6[ 0 ] - di_0[ 0 ] );
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            px_7[ 0 ] = px_1[ 0 ];
            py_7[ 0 ] = py_1[ 0 ];
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            px_2[ 0 ] = px_4[ 0 ];
            py_2[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            TF tmpx_1 = px_3[ 0 ];
            TF tmpy_1 = py_3[ 0 ];
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_0[ 0 ] + d_0_6 * ( px_0[ 0 ] - px_6[ 0 ] );
            py_5[ 0 ] = py_0[ 0 ] + d_0_6 * ( py_0[ 0 ] - py_6[ 0 ] );
            px_6[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_6[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_113: {
            // out: 1,0,0,0,0,0,0,0 cut: [0,1],1,2,3,4,5,6,7,[7,0]
            nodes_size = 9;
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_0_7 = di_0[ 0 ] / ( di_7[ 0 ] - di_0[ 0 ] );
            px_8[ 0 ] = px_0[ 0 ] + d_0_7 * ( px_0[ 0 ] - px_7[ 0 ] );
            py_8[ 0 ] = py_0[ 0 ] + d_0_7 * ( py_0[ 0 ] - py_7[ 0 ] );
            px_0[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_0[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_8[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_169: {
            // out: 1,0,0,0,0,0,0,0,0 cut: 2,3,4,5,6,7,8,[8,0],[0,1],1
            nodes_size = 10;
            TF d_8_0 = di_8[ 0 ] / ( di_0[ 0 ] - di_8[ 0 ] );
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            px[ 9 ] = px_1[ 0 ];
            py[ 9 ] = py_1[ 0 ];
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            px_2[ 0 ] = px_4[ 0 ];
            py_2[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_8[ 0 ];
            py_6[ 0 ] = py_8[ 0 ];
            TF tmpx_1 = px_3[ 0 ];
            TF tmpy_1 = py_3[ 0 ];
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_7[ 0 ];
            py_5[ 0 ] = py_7[ 0 ];
            px_7[ 0 ] = px_8[ 0 ] + d_8_0 * ( px_8[ 0 ] - px_0[ 0 ] );
            py_7[ 0 ] = py_8[ 0 ] + d_8_0 * ( py_8[ 0 ] - py_0[ 0 ] );
            px_8[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_8[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_8[ 0 ];
            pi_8[ 0 ] = cut_i[ num_cut ];
            pi[ 9 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_7[ 0 ];
            pi_7[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_206: {
            // out: 1,0,0,0,0,0,0,0,1 cut: [0,1],1,2,3,4,5,6,7,[7,8]
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_7_8 = di_7[ 0 ] / ( di_8[ 0 ] - di_7[ 0 ] );
            px_0[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_8[ 0 ] = px_7[ 0 ] + d_7_8 * ( px_7[ 0 ] - px_8[ 0 ] );
            py_8[ 0 ] = py_7[ 0 ] + d_7_8 * ( py_7[ 0 ] - py_8[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_8[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_142: {
            // out: 1,0,0,0,0,0,0,1 cut: [0,1],1,2,3,4,5,6,[6,7]
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            px_0[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_0[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_7[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_7[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_214: {
            // out: 1,0,0,0,0,0,0,1,1 cut: [0,1],1,2,3,4,5,6,[6,7]
            nodes_size = 8;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            px_0[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_7[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_7[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_93: {
            // out: 1,0,0,0,0,0,1 cut: [0,1],1,2,3,4,5,[5,6]
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_0[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_6[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_6[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_149: {
            // out: 1,0,0,0,0,0,1,1 cut: [0,1],1,2,3,4,5,[5,6]
            nodes_size = 7;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            px_0[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_6[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_6[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_221: {
            // out: 1,0,0,0,0,0,1,1,1 cut: [0,1],1,2,3,4,5,[5,6]
            nodes_size = 7;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            px_0[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_6[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_6[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_57: {
            // out: 1,0,0,0,0,1 cut: [0,1],1,2,3,4,[4,5]
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_0[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_5[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_5[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_99: {
            // out: 1,0,0,0,0,1,1 cut: [0,1],1,2,3,4,[4,5]
            nodes_size = 6;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_5[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_5[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_155: {
            // out: 1,0,0,0,0,1,1,1 cut: [0,1],1,2,3,4,[4,5]
            nodes_size = 6;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            px_0[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_5[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_5[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_227: {
            // out: 1,0,0,0,0,1,1,1,1 cut: [0,1],1,2,3,4,[4,5]
            nodes_size = 6;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            px_0[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_5[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_5[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_32: {
            // out: 1,0,0,0,1 cut: 3,[3,4],[0,1],1,2
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            TF tmpx_1 = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            TF tmpy_1 = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_1[ 0 ];
            py_3[ 0 ] = py_1[ 0 ];
            px_4[ 0 ] = px_2[ 0 ];
            py_4[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_62: {
            // out: 1,0,0,0,1,1 cut: 1,2,3,[3,4],[0,1]
            nodes_size = 5;
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            TF tmpx_1 = px_2[ 0 ];
            TF tmpy_1 = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_3[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_4[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_4[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_104: {
            // out: 1,0,0,0,1,1,1 cut: 1,2,3,[3,4],[0,1]
            nodes_size = 5;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            TF tmpx_1 = px_2[ 0 ];
            TF tmpy_1 = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_4[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_4[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_160: {
            // out: 1,0,0,0,1,1,1,1 cut: 1,2,3,[3,4],[0,1]
            nodes_size = 5;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            TF tmpx_1 = px_2[ 0 ];
            TF tmpy_1 = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_4[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_4[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_232: {
            // out: 1,0,0,0,1,1,1,1,1 cut: 1,2,3,[3,4],[0,1]
            nodes_size = 5;
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            TF tmpx_1 = px_2[ 0 ];
            TF tmpy_1 = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_3[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_4[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_4[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_16: {
            // out: 1,0,0,1 cut: [2,3],[0,1],1,2
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF tmpx_0 = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            TF tmpy_0 = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_2[ 0 ];
            py_3[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_1[ 0 ];
            py_2[ 0 ] = py_1[ 0 ];
            px_1[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_36: {
            // out: 1,0,0,1,1 cut: 1,2,[2,3],[0,1]
            nodes_size = 4;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            TF tmpx_1 = px_2[ 0 ];
            TF tmpy_1 = py_2[ 0 ];
            px_2[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_3[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_66: {
            // out: 1,0,0,1,1,1 cut: 1,2,[2,3],[0,1]
            nodes_size = 4;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            TF tmpx_1 = px_2[ 0 ];
            TF tmpy_1 = py_2[ 0 ];
            px_2[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_3[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_108: {
            // out: 1,0,0,1,1,1,1 cut: 1,2,[2,3],[0,1]
            nodes_size = 4;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            TF tmpx_1 = px_2[ 0 ];
            TF tmpy_1 = py_2[ 0 ];
            px_2[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_3[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_164: {
            // out: 1,0,0,1,1,1,1,1 cut: 2,[2,3],[0,1],1
            nodes_size = 4;
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            TF tmpx_1 = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            TF tmpy_1 = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_3[ 0 ] = px_1[ 0 ];
            py_3[ 0 ] = py_1[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_236: {
            // out: 1,0,0,1,1,1,1,1,1 cut: 2,[2,3],[0,1],1
            nodes_size = 4;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            TF tmpx_1 = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            TF tmpy_1 = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_3[ 0 ] = px_1[ 0 ];
            py_3[ 0 ] = py_1[ 0 ];
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_6: {
            // out: 1,0,1 cut: [0,1],1,[1,2]
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_0[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_19: {
            // out: 1,0,1,1 cut: [0,1],1,[1,2]
            nodes_size = 3;
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_39: {
            // out: 1,0,1,1,1 cut: 1,[1,2],[0,1]
            nodes_size = 3;
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            TF tmpx_1 = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            TF tmpy_1 = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_69: {
            // out: 1,0,1,1,1,1 cut: 1,[1,2],[0,1]
            nodes_size = 3;
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            TF tmpx_1 = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            TF tmpy_1 = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_111: {
            // out: 1,0,1,1,1,1,1 cut: 1,[1,2],[0,1]
            nodes_size = 3;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            TF tmpx_1 = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            TF tmpy_1 = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_167: {
            // out: 1,0,1,1,1,1,1,1 cut: 1,[1,2],[0,1]
            nodes_size = 3;
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF d_0_1 = di_0[ 0 ] / ( di_1[ 0 ] - di_0[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            TF tmpx_1 = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            TF tmpy_1 = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ] + d_0_1 * ( px_0[ 0 ] - px_1[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_1 * ( py_0[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_239: {
            // out: 1,0,1,1,1,1,1,1,1 cut: 1,[1,2],[0,1]
            nodes_size = 3;
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            TF d_1_0 = di_1[ 0 ] / ( di_0[ 0 ] - di_1[ 0 ] );
            TF tmpx_0 = px_1[ 0 ];
            TF tmpy_0 = py_1[ 0 ];
            TF tmpx_1 = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            TF tmpy_1 = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_0 * ( px_1[ 0 ] - px_0[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_0 * ( py_1[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_1[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_4: {
            // out: 1,1,0 cut: [2,0],[1,2],2
            TF d_0_2 = di_0[ 0 ] / ( di_2[ 0 ] - di_0[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_0[ 0 ] + d_0_2 * ( px_0[ 0 ] - px_2[ 0 ] );
            py_0[ 0 ] = py_0[ 0 ] + d_0_2 * ( py_0[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_1[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_11: {
            // out: 1,1,0,0 cut: 2,3,[3,0],[1,2]
            TF d_0_3 = di_0[ 0 ] / ( di_3[ 0 ] - di_0[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            TF tmpx_1 = px_3[ 0 ];
            TF tmpy_1 = py_3[ 0 ];
            TF tmpx_2 = px_0[ 0 ] + d_0_3 * ( px_0[ 0 ] - px_3[ 0 ] );
            TF tmpy_2 = py_0[ 0 ] + d_0_3 * ( py_0[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            px_2[ 0 ] = tmpx_2;
            py_2[ 0 ] = tmpy_2;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_23: {
            // out: 1,1,0,0,0 cut: [1,2],2,3,4,[4,0]
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF d_0_4 = di_0[ 0 ] / ( di_4[ 0 ] - di_0[ 0 ] );
            TF tmpx_0 = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            TF tmpy_0 = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = px_2[ 0 ];
            py_1[ 0 ] = py_2[ 0 ];
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_0[ 0 ] + d_0_4 * ( px_0[ 0 ] - px_4[ 0 ] );
            py_4[ 0 ] = py_0[ 0 ] + d_0_4 * ( py_0[ 0 ] - py_4[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_1[ 0 ] = pi_2[ 0 ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_43: {
            // out: 1,1,0,0,0,0 cut: 2,3,4,5,[5,0],[1,2]
            TF d_5_0 = di_5[ 0 ] / ( di_0[ 0 ] - di_5[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            TF tmpx_1 = px_3[ 0 ];
            TF tmpy_1 = py_3[ 0 ];
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            TF tmpx_2 = px_4[ 0 ];
            TF tmpy_2 = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ] + d_5_0 * ( px_5[ 0 ] - px_0[ 0 ] );
            py_4[ 0 ] = py_5[ 0 ] + d_5_0 * ( py_5[ 0 ] - py_0[ 0 ] );
            px_5[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_5[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            px_2[ 0 ] = tmpx_2;
            py_2[ 0 ] = tmpy_2;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = pi_5[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_73: {
            // out: 1,1,0,0,0,0,0 cut: 2,3,4,5,6,[6,0],[1,2]
            TF d_6_0 = di_6[ 0 ] / ( di_0[ 0 ] - di_6[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            TF tmpx_1 = px_3[ 0 ];
            TF tmpy_1 = py_3[ 0 ];
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ] + d_6_0 * ( px_6[ 0 ] - px_0[ 0 ] );
            py_5[ 0 ] = py_6[ 0 ] + d_6_0 * ( py_6[ 0 ] - py_0[ 0 ] );
            TF tmpx_2 = px_4[ 0 ];
            TF tmpy_2 = py_4[ 0 ];
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_6[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            px_2[ 0 ] = tmpx_2;
            py_2[ 0 ] = tmpy_2;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = pi_5[ 0 ];
            pi_4[ 0 ] = pi_6[ 0 ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_115: {
            // out: 1,1,0,0,0,0,0,0 cut: 2,3,4,5,6,7,[7,0],[1,2]
            TF d_0_7 = di_0[ 0 ] / ( di_7[ 0 ] - di_0[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            TF tmpx_1 = px_3[ 0 ];
            TF tmpy_1 = py_3[ 0 ];
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_7[ 0 ];
            py_5[ 0 ] = py_7[ 0 ];
            TF tmpx_2 = px_4[ 0 ];
            TF tmpy_2 = py_4[ 0 ];
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_0[ 0 ] + d_0_7 * ( px_0[ 0 ] - px_7[ 0 ] );
            py_6[ 0 ] = py_0[ 0 ] + d_0_7 * ( py_0[ 0 ] - py_7[ 0 ] );
            px_7[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_7[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            px_2[ 0 ] = tmpx_2;
            py_2[ 0 ] = tmpy_2;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = pi_5[ 0 ];
            pi_4[ 0 ] = pi_6[ 0 ];
            pi_5[ 0 ] = pi_7[ 0 ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            pi_7[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_171: {
            // out: 1,1,0,0,0,0,0,0,0 cut: 2,3,4,5,6,7,8,[8,0],[1,2]
            TF d_0_8 = di_0[ 0 ] / ( di_8[ 0 ] - di_0[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            TF tmpx_0 = px_2[ 0 ];
            TF tmpy_0 = py_2[ 0 ];
            TF tmpx_1 = px_3[ 0 ];
            TF tmpy_1 = py_3[ 0 ];
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_7[ 0 ];
            py_5[ 0 ] = py_7[ 0 ];
            px_7[ 0 ] = px_0[ 0 ] + d_0_8 * ( px_0[ 0 ] - px_8[ 0 ] );
            py_7[ 0 ] = py_0[ 0 ] + d_0_8 * ( py_0[ 0 ] - py_8[ 0 ] );
            TF tmpx_2 = px_4[ 0 ];
            TF tmpy_2 = py_4[ 0 ];
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_8[ 0 ];
            py_6[ 0 ] = py_8[ 0 ];
            px_8[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_8[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            px_1[ 0 ] = tmpx_1;
            py_1[ 0 ] = tmpy_1;
            px_2[ 0 ] = tmpx_2;
            py_2[ 0 ] = tmpy_2;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = pi_5[ 0 ];
            pi_4[ 0 ] = pi_6[ 0 ];
            pi_5[ 0 ] = pi_7[ 0 ];
            pi_6[ 0 ] = pi_8[ 0 ];
            pi_7[ 0 ] = cut_i[ num_cut ];
            pi_8[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_207: {
            // out: 1,1,0,0,0,0,0,0,1 cut: 4,5,6,7,[7,8],[1,2],2,3
            nodes_size = 8;
            TF d_8_7 = di_8[ 0 ] / ( di_7[ 0 ] - di_8[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_4[ 0 ];
            py_0[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_8[ 0 ] + d_8_7 * ( px_8[ 0 ] - px_7[ 0 ] );
            py_4[ 0 ] = py_8[ 0 ] + d_8_7 * ( py_8[ 0 ] - py_7[ 0 ] );
            TF tmpx_0 = px_5[ 0 ];
            TF tmpy_0 = py_5[ 0 ];
            px_5[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_5[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            TF tmpx_1 = px_6[ 0 ];
            TF tmpy_1 = py_6[ 0 ];
            px_6[ 0 ] = px_2[ 0 ];
            py_6[ 0 ] = py_2[ 0 ];
            TF tmpx_2 = px_7[ 0 ];
            TF tmpy_2 = py_7[ 0 ];
            px_7[ 0 ] = px_3[ 0 ];
            py_7[ 0 ] = py_3[ 0 ];
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            px_3[ 0 ] = tmpx_2;
            py_3[ 0 ] = tmpy_2;
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_1[ 0 ] = pi_5[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            TF tmpi_0 = pi_6[ 0 ];
            pi_6[ 0 ] = pi_2[ 0 ];
            TF tmpi_1 = pi_7[ 0 ];
            pi_7[ 0 ] = pi_3[ 0 ];
            px_2[ 0 ] = tmpx_0;
            py_2[ 0 ] = tmpy_0;
            px_3[ 0 ] = tmpx_1;
            py_3[ 0 ] = tmpy_1;
            continue;
        }
        
        case_143: {
            // out: 1,1,0,0,0,0,0,1 cut: 2,3,4,5,6,[6,7],[1,2]
            nodes_size = 7;
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_5[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            TF tmpx_1 = px_4[ 0 ];
            TF tmpy_1 = py_4[ 0 ];
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_6[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = pi_5[ 0 ];
            pi_4[ 0 ] = pi_6[ 0 ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_215: {
            // out: 1,1,0,0,0,0,0,1,1 cut: 2,3,4,5,6,[6,7],[1,2]
            nodes_size = 7;
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_5[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            TF tmpx_1 = px_4[ 0 ];
            TF tmpy_1 = py_4[ 0 ];
            px_4[ 0 ] = px_6[ 0 ];
            py_4[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_6[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = pi_5[ 0 ];
            pi_4[ 0 ] = pi_6[ 0 ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            pi_6[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_94: {
            // out: 1,1,0,0,0,0,1 cut: 3,4,5,[5,6],[1,2],2
            nodes_size = 6;
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_3[ 0 ];
            py_0[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_3[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            TF tmpx_0 = px_4[ 0 ];
            TF tmpy_0 = py_4[ 0 ];
            px_4[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_4[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            TF tmpx_1 = px_5[ 0 ];
            TF tmpy_1 = py_5[ 0 ];
            px_5[ 0 ] = px_2[ 0 ];
            py_5[ 0 ] = py_2[ 0 ];
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_1[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            TF tmpi_0 = pi_5[ 0 ];
            pi_5[ 0 ] = pi_2[ 0 ];
            px_2[ 0 ] = tmpx_0;
            py_2[ 0 ] = tmpy_0;
            continue;
        }
        
        case_150: {
            // out: 1,1,0,0,0,0,1,1 cut: 3,4,5,[5,6],[1,2],2
            nodes_size = 6;
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_3[ 0 ];
            py_0[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_3[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            TF tmpx_0 = px_4[ 0 ];
            TF tmpy_0 = py_4[ 0 ];
            px_4[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_4[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            TF tmpx_1 = px_5[ 0 ];
            TF tmpy_1 = py_5[ 0 ];
            px_5[ 0 ] = px_2[ 0 ];
            py_5[ 0 ] = py_2[ 0 ];
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_1[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            TF tmpi_0 = pi_5[ 0 ];
            pi_5[ 0 ] = pi_2[ 0 ];
            px_2[ 0 ] = tmpx_0;
            py_2[ 0 ] = tmpy_0;
            continue;
        }
        
        case_222: {
            // out: 1,1,0,0,0,0,1,1,1 cut: 3,4,5,[5,6],[1,2],2
            nodes_size = 6;
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_3[ 0 ];
            py_0[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_3[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            TF tmpx_0 = px_4[ 0 ];
            TF tmpy_0 = py_4[ 0 ];
            px_4[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_4[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            TF tmpx_1 = px_5[ 0 ];
            TF tmpy_1 = py_5[ 0 ];
            px_5[ 0 ] = px_2[ 0 ];
            py_5[ 0 ] = py_2[ 0 ];
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_1[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            TF tmpi_0 = pi_5[ 0 ];
            pi_5[ 0 ] = pi_2[ 0 ];
            px_2[ 0 ] = tmpx_0;
            py_2[ 0 ] = tmpy_0;
            continue;
        }
        
        case_58: {
            // out: 1,1,0,0,0,1 cut: 2,3,4,[4,5],[1,2]
            nodes_size = 5;
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            TF tmpx_1 = px_4[ 0 ];
            TF tmpy_1 = py_4[ 0 ];
            px_4[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_4[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_100: {
            // out: 1,1,0,0,0,1,1 cut: 2,3,4,[4,5],[1,2]
            nodes_size = 5;
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            TF tmpx_1 = px_4[ 0 ];
            TF tmpy_1 = py_4[ 0 ];
            px_4[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_4[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_156: {
            // out: 1,1,0,0,0,1,1,1 cut: 2,3,4,[4,5],[1,2]
            nodes_size = 5;
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            px_3[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_3[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            TF tmpx_1 = px_4[ 0 ];
            TF tmpy_1 = py_4[ 0 ];
            px_4[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_4[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_228: {
            // out: 1,1,0,0,0,1,1,1,1 cut: 2,3,4,[4,5],[1,2]
            nodes_size = 5;
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            px_3[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_3[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            TF tmpx_1 = px_4[ 0 ];
            TF tmpy_1 = py_4[ 0 ];
            px_4[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_4[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_33: {
            // out: 1,1,0,0,1 cut: [3,4],[1,2],2,3
            nodes_size = 4;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_0[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_1[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_1[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_63: {
            // out: 1,1,0,0,1,1 cut: 2,3,[3,4],[1,2]
            nodes_size = 4;
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            TF tmpx_1 = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            TF tmpy_1 = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_3[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_105: {
            // out: 1,1,0,0,1,1,1 cut: 2,3,[3,4],[1,2]
            nodes_size = 4;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            TF tmpx_1 = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            TF tmpy_1 = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_161: {
            // out: 1,1,0,0,1,1,1,1 cut: 2,3,[3,4],[1,2]
            nodes_size = 4;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            TF tmpx_1 = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            TF tmpy_1 = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_233: {
            // out: 1,1,0,0,1,1,1,1,1 cut: 2,3,[3,4],[1,2]
            nodes_size = 4;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_2_1 = di_2[ 0 ] / ( di_1[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            TF tmpx_1 = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            TF tmpy_1 = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_2[ 0 ] + d_2_1 * ( px_2[ 0 ] - px_1[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_1 * ( py_2[ 0 ] - py_1[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            px_2[ 0 ] = tmpx_1;
            py_2[ 0 ] = tmpy_1;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_17: {
            // out: 1,1,0,1 cut: [2,3],[1,2],2
            nodes_size = 3;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_0[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_1[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_1[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_37: {
            // out: 1,1,0,1,1 cut: [2,3],[1,2],2
            nodes_size = 3;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_0[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_1[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_1[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_67: {
            // out: 1,1,0,1,1,1 cut: 2,[2,3],[1,2]
            nodes_size = 3;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            TF tmpy_0 = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_109: {
            // out: 1,1,0,1,1,1,1 cut: 2,[2,3],[1,2]
            nodes_size = 3;
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            TF tmpy_0 = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_165: {
            // out: 1,1,0,1,1,1,1,1 cut: 2,[2,3],[1,2]
            nodes_size = 3;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            TF tmpy_0 = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_237: {
            // out: 1,1,0,1,1,1,1,1,1 cut: 2,[2,3],[1,2]
            nodes_size = 3;
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            TF d_1_2 = di_1[ 0 ] / ( di_2[ 0 ] - di_1[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ];
            py_0[ 0 ] = py_2[ 0 ];
            TF tmpx_0 = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            TF tmpy_0 = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_1[ 0 ] + d_1_2 * ( px_1[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_1[ 0 ] + d_1_2 * ( py_1[ 0 ] - py_2[ 0 ] );
            px_1[ 0 ] = tmpx_0;
            py_1[ 0 ] = tmpy_0;
            pi_0[ 0 ] = pi_2[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_14: {
            // out: 1,1,1,0 cut: 3,[3,0],[2,3]
            nodes_size = 3;
            TF d_3_0 = di_3[ 0 ] / ( di_0[ 0 ] - di_3[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_1[ 0 ] = px_3[ 0 ] + d_3_0 * ( px_3[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_3[ 0 ] + d_3_0 * ( py_3[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = px_3[ 0 ];
            py_0[ 0 ] = py_3[ 0 ];
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_26: {
            // out: 1,1,1,0,0 cut: 3,4,[4,0],[2,3]
            nodes_size = 4;
            TF d_0_4 = di_0[ 0 ] / ( di_4[ 0 ] - di_0[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_1[ 0 ] = px_4[ 0 ];
            py_1[ 0 ] = py_4[ 0 ];
            TF tmpx_0 = px_3[ 0 ];
            TF tmpy_0 = py_3[ 0 ];
            px_3[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_3[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            px_2[ 0 ] = px_0[ 0 ] + d_0_4 * ( px_0[ 0 ] - px_4[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_4 * ( py_0[ 0 ] - py_4[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_1[ 0 ] = pi_4[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_46: {
            // out: 1,1,1,0,0,0 cut: 5,[5,0],[2,3],3,4
            nodes_size = 5;
            TF d_0_5 = di_0[ 0 ] / ( di_5[ 0 ] - di_0[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_5 * ( px_0[ 0 ] - px_5[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_5 * ( py_0[ 0 ] - py_5[ 0 ] );
            px_0[ 0 ] = px_5[ 0 ];
            py_0[ 0 ] = py_5[ 0 ];
            px_2[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_76: {
            // out: 1,1,1,0,0,0,0 cut: [6,0],[2,3],3,4,5,6
            nodes_size = 6;
            TF d_6_0 = di_6[ 0 ] / ( di_0[ 0 ] - di_6[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ] + d_6_0 * ( px_6[ 0 ] - px_0[ 0 ] );
            py_0[ 0 ] = py_6[ 0 ] + d_6_0 * ( py_6[ 0 ] - py_0[ 0 ] );
            px_1[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_1[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ];
            py_2[ 0 ] = py_3[ 0 ];
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = pi_3[ 0 ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_6[ 0 ];
            continue;
        }
        
        case_118: {
            // out: 1,1,1,0,0,0,0,0 cut: 7,[7,0],[2,3],3,4,5,6
            nodes_size = 7;
            TF d_0_7 = di_0[ 0 ] / ( di_7[ 0 ] - di_0[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_7 * ( px_0[ 0 ] - px_7[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_7 * ( py_0[ 0 ] - py_7[ 0 ] );
            px_0[ 0 ] = px_7[ 0 ];
            py_0[ 0 ] = py_7[ 0 ];
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_7[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_174: {
            // out: 1,1,1,0,0,0,0,0,0 cut: 8,[8,0],[2,3],3,4,5,6,7
            nodes_size = 8;
            TF d_0_8 = di_0[ 0 ] / ( di_8[ 0 ] - di_0[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_8 * ( px_0[ 0 ] - px_8[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_8 * ( py_0[ 0 ] - py_8[ 0 ] );
            px_0[ 0 ] = px_8[ 0 ];
            py_0[ 0 ] = py_8[ 0 ];
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_8[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_208: {
            // out: 1,1,1,0,0,0,0,0,1 cut: 7,[7,8],[2,3],3,4,5,6
            nodes_size = 7;
            TF d_7_8 = di_7[ 0 ] / ( di_8[ 0 ] - di_7[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_7[ 0 ];
            py_0[ 0 ] = py_7[ 0 ];
            px_1[ 0 ] = px_7[ 0 ] + d_7_8 * ( px_7[ 0 ] - px_8[ 0 ] );
            py_1[ 0 ] = py_7[ 0 ] + d_7_8 * ( py_7[ 0 ] - py_8[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_7[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_144: {
            // out: 1,1,1,0,0,0,0,1 cut: 6,[6,7],[2,3],3,4,5
            nodes_size = 6;
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_1[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_1[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            px_2[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_216: {
            // out: 1,1,1,0,0,0,0,1,1 cut: 6,[6,7],[2,3],3,4,5
            nodes_size = 6;
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_1[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_1[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_95: {
            // out: 1,1,1,0,0,0,1 cut: [2,3],3,4,5,[5,6]
            nodes_size = 5;
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_0[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_0[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_1[ 0 ] = px_3[ 0 ];
            py_1[ 0 ] = py_3[ 0 ];
            px_2[ 0 ] = px_4[ 0 ];
            py_2[ 0 ] = py_4[ 0 ];
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            px_4[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_4[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            pi_0[ 0 ] = cut_i[ num_cut ];
            pi_1[ 0 ] = pi_3[ 0 ];
            pi_2[ 0 ] = pi_4[ 0 ];
            pi_3[ 0 ] = pi_5[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_151: {
            // out: 1,1,1,0,0,0,1,1 cut: 5,[5,6],[2,3],3,4
            nodes_size = 5;
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_5[ 0 ];
            py_0[ 0 ] = py_5[ 0 ];
            px_1[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_1[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_223: {
            // out: 1,1,1,0,0,0,1,1,1 cut: 3,4,5,[5,6],[2,3]
            nodes_size = 5;
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_3[ 0 ];
            py_0[ 0 ] = py_3[ 0 ];
            px_1[ 0 ] = px_4[ 0 ];
            py_1[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_4[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_5[ 0 ];
            py_2[ 0 ] = py_5[ 0 ];
            px_3[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_3[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_1[ 0 ] = pi_4[ 0 ];
            pi_2[ 0 ] = pi_5[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_59: {
            // out: 1,1,1,0,0,1 cut: 3,4,[4,5],[2,3]
            nodes_size = 4;
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_3[ 0 ];
            py_0[ 0 ] = py_3[ 0 ];
            px_1[ 0 ] = px_4[ 0 ];
            py_1[ 0 ] = py_4[ 0 ];
            px_3[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_3[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_2[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_1[ 0 ] = pi_4[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_101: {
            // out: 1,1,1,0,0,1,1 cut: 4,[4,5],[2,3],3
            nodes_size = 4;
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_4[ 0 ];
            py_0[ 0 ] = py_4[ 0 ];
            px_1[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_1[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_157: {
            // out: 1,1,1,0,0,1,1,1 cut: 4,[4,5],[2,3],3
            nodes_size = 4;
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_4[ 0 ];
            py_0[ 0 ] = py_4[ 0 ];
            px_1[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_1[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_229: {
            // out: 1,1,1,0,0,1,1,1,1 cut: 4,[4,5],[2,3],3
            nodes_size = 4;
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            TF d_2_3 = di_2[ 0 ] / ( di_3[ 0 ] - di_2[ 0 ] );
            px_0[ 0 ] = px_4[ 0 ];
            py_0[ 0 ] = py_4[ 0 ];
            px_1[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_1[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            px_2[ 0 ] = px_2[ 0 ] + d_2_3 * ( px_2[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_2[ 0 ] + d_2_3 * ( py_2[ 0 ] - py_3[ 0 ] );
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_34: {
            // out: 1,1,1,0,1 cut: 3,[3,4],[2,3]
            nodes_size = 3;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_3[ 0 ];
            py_0[ 0 ] = py_3[ 0 ];
            px_1[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_1[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_64: {
            // out: 1,1,1,0,1,1 cut: 3,[3,4],[2,3]
            nodes_size = 3;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_3[ 0 ];
            py_0[ 0 ] = py_3[ 0 ];
            px_1[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_1[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_106: {
            // out: 1,1,1,0,1,1,1 cut: 3,[3,4],[2,3]
            nodes_size = 3;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_3[ 0 ];
            py_0[ 0 ] = py_3[ 0 ];
            px_1[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_1[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_162: {
            // out: 1,1,1,0,1,1,1,1 cut: 3,[3,4],[2,3]
            nodes_size = 3;
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_3[ 0 ];
            py_0[ 0 ] = py_3[ 0 ];
            px_1[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_1[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_234: {
            // out: 1,1,1,0,1,1,1,1,1 cut: 3,[3,4],[2,3]
            nodes_size = 3;
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            TF d_3_2 = di_3[ 0 ] / ( di_2[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_3[ 0 ];
            py_0[ 0 ] = py_3[ 0 ];
            px_1[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_1[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_2 * ( px_3[ 0 ] - px_2[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_2 * ( py_3[ 0 ] - py_2[ 0 ] );
            pi_0[ 0 ] = pi_3[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_30: {
            // out: 1,1,1,1,0 cut: 4,[4,0],[3,4]
            nodes_size = 3;
            TF d_4_0 = di_4[ 0 ] / ( di_0[ 0 ] - di_4[ 0 ] );
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            px_1[ 0 ] = px_4[ 0 ] + d_4_0 * ( px_4[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_4[ 0 ] + d_4_0 * ( py_4[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = px_4[ 0 ];
            py_0[ 0 ] = py_4[ 0 ];
            px_2[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_50: {
            // out: 1,1,1,1,0,0 cut: 5,[5,0],[3,4],4
            nodes_size = 4;
            TF d_5_0 = di_5[ 0 ] / ( di_0[ 0 ] - di_5[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_1[ 0 ] = px_5[ 0 ] + d_5_0 * ( px_5[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_5[ 0 ] + d_5_0 * ( py_5[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = px_5[ 0 ];
            py_0[ 0 ] = py_5[ 0 ];
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            continue;
        }
        
        case_80: {
            // out: 1,1,1,1,0,0,0 cut: 6,[6,0],[3,4],4,5
            nodes_size = 5;
            TF d_6_0 = di_6[ 0 ] / ( di_0[ 0 ] - di_6[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_1[ 0 ] = px_6[ 0 ] + d_6_0 * ( px_6[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_6[ 0 ] + d_6_0 * ( py_6[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            continue;
        }
        
        case_122: {
            // out: 1,1,1,1,0,0,0,0 cut: 4,5,6,7,[7,0],[3,4]
            nodes_size = 6;
            TF d_0_7 = di_0[ 0 ] / ( di_7[ 0 ] - di_0[ 0 ] );
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            px_1[ 0 ] = px_5[ 0 ];
            py_1[ 0 ] = py_5[ 0 ];
            px_2[ 0 ] = px_6[ 0 ];
            py_2[ 0 ] = py_6[ 0 ];
            px_5[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_5[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_7[ 0 ];
            py_3[ 0 ] = py_7[ 0 ];
            TF tmpx_0 = px_4[ 0 ];
            TF tmpy_0 = py_4[ 0 ];
            px_4[ 0 ] = px_0[ 0 ] + d_0_7 * ( px_0[ 0 ] - px_7[ 0 ] );
            py_4[ 0 ] = py_0[ 0 ] + d_0_7 * ( py_0[ 0 ] - py_7[ 0 ] );
            px_0[ 0 ] = tmpx_0;
            py_0[ 0 ] = tmpy_0;
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_1[ 0 ] = pi_5[ 0 ];
            pi_2[ 0 ] = pi_6[ 0 ];
            pi_3[ 0 ] = pi_7[ 0 ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            pi_5[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_178: {
            // out: 1,1,1,1,0,0,0,0,0 cut: 8,[8,0],[3,4],4,5,6,7
            nodes_size = 7;
            TF d_8_0 = di_8[ 0 ] / ( di_0[ 0 ] - di_8[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_1[ 0 ] = px_8[ 0 ] + d_8_0 * ( px_8[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_8[ 0 ] + d_8_0 * ( py_8[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = px_8[ 0 ];
            py_0[ 0 ] = py_8[ 0 ];
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            px_6[ 0 ] = px_7[ 0 ];
            py_6[ 0 ] = py_7[ 0 ];
            pi_0[ 0 ] = pi_8[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_6[ 0 ];
            pi_6[ 0 ] = pi_7[ 0 ];
            continue;
        }
        
        case_209: {
            // out: 1,1,1,1,0,0,0,0,1 cut: 7,[7,8],[3,4],4,5,6
            nodes_size = 6;
            TF d_7_8 = di_7[ 0 ] / ( di_8[ 0 ] - di_7[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_7[ 0 ];
            py_0[ 0 ] = py_7[ 0 ];
            px_1[ 0 ] = px_7[ 0 ] + d_7_8 * ( px_7[ 0 ] - px_8[ 0 ] );
            py_1[ 0 ] = py_7[ 0 ] + d_7_8 * ( py_7[ 0 ] - py_8[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            pi_0[ 0 ] = pi_7[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_6[ 0 ];
            continue;
        }
        
        case_145: {
            // out: 1,1,1,1,0,0,0,1 cut: 6,[6,7],[3,4],4,5
            nodes_size = 5;
            TF d_7_6 = di_7[ 0 ] / ( di_6[ 0 ] - di_7[ 0 ] );
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_1[ 0 ] = px_7[ 0 ] + d_7_6 * ( px_7[ 0 ] - px_6[ 0 ] );
            py_1[ 0 ] = py_7[ 0 ] + d_7_6 * ( py_7[ 0 ] - py_6[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            continue;
        }
        
        case_217: {
            // out: 1,1,1,1,0,0,0,1,1 cut: 6,[6,7],[3,4],4,5
            nodes_size = 5;
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_1[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_1[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            pi_4[ 0 ] = pi_5[ 0 ];
            continue;
        }
        
        case_96: {
            // out: 1,1,1,1,0,0,1 cut: 5,[5,6],[3,4],4
            nodes_size = 4;
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_5[ 0 ];
            py_0[ 0 ] = py_5[ 0 ];
            px_1[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_1[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            continue;
        }
        
        case_152: {
            // out: 1,1,1,1,0,0,1,1 cut: 5,[5,6],[3,4],4
            nodes_size = 4;
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_5[ 0 ];
            py_0[ 0 ] = py_5[ 0 ];
            px_1[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_1[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            continue;
        }
        
        case_224: {
            // out: 1,1,1,1,0,0,1,1,1 cut: 5,[5,6],[3,4],4
            nodes_size = 4;
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_5[ 0 ];
            py_0[ 0 ] = py_5[ 0 ];
            px_1[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_1[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ];
            py_3[ 0 ] = py_4[ 0 ];
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_4[ 0 ];
            continue;
        }
        
        case_60: {
            // out: 1,1,1,1,0,1 cut: 4,[4,5],[3,4]
            nodes_size = 3;
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_4[ 0 ];
            py_0[ 0 ] = py_4[ 0 ];
            px_1[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_1[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_102: {
            // out: 1,1,1,1,0,1,1 cut: 4,[4,5],[3,4]
            nodes_size = 3;
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_4[ 0 ];
            py_0[ 0 ] = py_4[ 0 ];
            px_1[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_1[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_158: {
            // out: 1,1,1,1,0,1,1,1 cut: 4,[4,5],[3,4]
            nodes_size = 3;
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            TF d_4_3 = di_4[ 0 ] / ( di_3[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_4[ 0 ];
            py_0[ 0 ] = py_4[ 0 ];
            px_1[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_1[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_3 * ( px_4[ 0 ] - px_3[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_3 * ( py_4[ 0 ] - py_3[ 0 ] );
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_230: {
            // out: 1,1,1,1,0,1,1,1,1 cut: 4,[4,5],[3,4]
            nodes_size = 3;
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            TF d_3_4 = di_3[ 0 ] / ( di_4[ 0 ] - di_3[ 0 ] );
            px_0[ 0 ] = px_4[ 0 ];
            py_0[ 0 ] = py_4[ 0 ];
            px_1[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_1[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            px_2[ 0 ] = px_3[ 0 ] + d_3_4 * ( px_3[ 0 ] - px_4[ 0 ] );
            py_2[ 0 ] = py_3[ 0 ] + d_3_4 * ( py_3[ 0 ] - py_4[ 0 ] );
            pi_0[ 0 ] = pi_4[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_55: {
            // out: 1,1,1,1,1,0 cut: 5,[5,0],[4,5]
            nodes_size = 3;
            TF d_5_0 = di_5[ 0 ] / ( di_0[ 0 ] - di_5[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_1[ 0 ] = px_5[ 0 ] + d_5_0 * ( px_5[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_5[ 0 ] + d_5_0 * ( py_5[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = px_5[ 0 ];
            py_0[ 0 ] = py_5[ 0 ];
            px_2[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_85: {
            // out: 1,1,1,1,1,0,0 cut: 6,[6,0],[4,5],5
            nodes_size = 4;
            TF d_0_6 = di_0[ 0 ] / ( di_6[ 0 ] - di_0[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_6 * ( px_0[ 0 ] - px_6[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_6 * ( py_0[ 0 ] - py_6[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_2[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_5[ 0 ];
            continue;
        }
        
        case_127: {
            // out: 1,1,1,1,1,0,0,0 cut: 6,7,[7,0],[4,5],5
            nodes_size = 5;
            TF d_7_0 = di_7[ 0 ] / ( di_0[ 0 ] - di_7[ 0 ] );
            TF d_5_4 = di_5[ 0 ] / ( di_4[ 0 ] - di_5[ 0 ] );
            px_1[ 0 ] = px_7[ 0 ];
            py_1[ 0 ] = py_7[ 0 ];
            px_2[ 0 ] = px_7[ 0 ] + d_7_0 * ( px_7[ 0 ] - px_0[ 0 ] );
            py_2[ 0 ] = py_7[ 0 ] + d_7_0 * ( py_7[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_3[ 0 ] = px_5[ 0 ] + d_5_4 * ( px_5[ 0 ] - px_4[ 0 ] );
            py_3[ 0 ] = py_5[ 0 ] + d_5_4 * ( py_5[ 0 ] - py_4[ 0 ] );
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = pi_7[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_5[ 0 ];
            continue;
        }
        
        case_183: {
            // out: 1,1,1,1,1,0,0,0,0 cut: 7,8,[8,0],[4,5],5,6
            nodes_size = 6;
            TF d_0_8 = di_0[ 0 ] / ( di_8[ 0 ] - di_0[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_1[ 0 ] = px_8[ 0 ];
            py_1[ 0 ] = py_8[ 0 ];
            px_2[ 0 ] = px_0[ 0 ] + d_0_8 * ( px_0[ 0 ] - px_8[ 0 ] );
            py_2[ 0 ] = py_0[ 0 ] + d_0_8 * ( py_0[ 0 ] - py_8[ 0 ] );
            px_0[ 0 ] = px_7[ 0 ];
            py_0[ 0 ] = py_7[ 0 ];
            px_3[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            px_5[ 0 ] = px_6[ 0 ];
            py_5[ 0 ] = py_6[ 0 ];
            pi_0[ 0 ] = pi_7[ 0 ];
            pi_1[ 0 ] = pi_8[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_5[ 0 ];
            pi_5[ 0 ] = pi_6[ 0 ];
            continue;
        }
        
        case_210: {
            // out: 1,1,1,1,1,0,0,0,1 cut: 6,7,[7,8],[4,5],5
            nodes_size = 5;
            TF d_7_8 = di_7[ 0 ] / ( di_8[ 0 ] - di_7[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_1[ 0 ] = px_7[ 0 ];
            py_1[ 0 ] = py_7[ 0 ];
            px_2[ 0 ] = px_7[ 0 ] + d_7_8 * ( px_7[ 0 ] - px_8[ 0 ] );
            py_2[ 0 ] = py_7[ 0 ] + d_7_8 * ( py_7[ 0 ] - py_8[ 0 ] );
            px_3[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_3[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_4[ 0 ] = px_5[ 0 ];
            py_4[ 0 ] = py_5[ 0 ];
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = pi_7[ 0 ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = pi_5[ 0 ];
            continue;
        }
        
        case_146: {
            // out: 1,1,1,1,1,0,0,1 cut: 6,[6,7],[4,5],5
            nodes_size = 4;
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_1[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_1[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_5[ 0 ];
            continue;
        }
        
        case_218: {
            // out: 1,1,1,1,1,0,0,1,1 cut: 6,[6,7],[4,5],5
            nodes_size = 4;
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_1[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_1[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            px_3[ 0 ] = px_5[ 0 ];
            py_3[ 0 ] = py_5[ 0 ];
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_5[ 0 ];
            continue;
        }
        
        case_97: {
            // out: 1,1,1,1,1,0,1 cut: 5,[5,6],[4,5]
            nodes_size = 3;
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_5[ 0 ];
            py_0[ 0 ] = py_5[ 0 ];
            px_1[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_1[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_153: {
            // out: 1,1,1,1,1,0,1,1 cut: 5,[5,6],[4,5]
            nodes_size = 3;
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_5[ 0 ];
            py_0[ 0 ] = py_5[ 0 ];
            px_1[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_1[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_225: {
            // out: 1,1,1,1,1,0,1,1,1 cut: 5,[5,6],[4,5]
            nodes_size = 3;
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            TF d_4_5 = di_4[ 0 ] / ( di_5[ 0 ] - di_4[ 0 ] );
            px_0[ 0 ] = px_5[ 0 ];
            py_0[ 0 ] = py_5[ 0 ];
            px_1[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_1[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_2[ 0 ] = px_4[ 0 ] + d_4_5 * ( px_4[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_4[ 0 ] + d_4_5 * ( py_4[ 0 ] - py_5[ 0 ] );
            pi_0[ 0 ] = pi_5[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_91: {
            // out: 1,1,1,1,1,1,0 cut: 6,[6,0],[5,6]
            nodes_size = 3;
            TF d_0_6 = di_0[ 0 ] / ( di_6[ 0 ] - di_0[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_6 * ( px_0[ 0 ] - px_6[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_6 * ( py_0[ 0 ] - py_6[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_2[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_2[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_133: {
            // out: 1,1,1,1,1,1,0,0 cut: 7,[7,0],[5,6],6
            nodes_size = 4;
            TF d_7_0 = di_7[ 0 ] / ( di_0[ 0 ] - di_7[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_1[ 0 ] = px_7[ 0 ] + d_7_0 * ( px_7[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_7[ 0 ] + d_7_0 * ( py_7[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = px_7[ 0 ];
            py_0[ 0 ] = py_7[ 0 ];
            px_2[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_2[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_3[ 0 ] = px_6[ 0 ];
            py_3[ 0 ] = py_6[ 0 ];
            pi_0[ 0 ] = pi_7[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_6[ 0 ];
            continue;
        }
        
        case_189: {
            // out: 1,1,1,1,1,1,0,0,0 cut: 6,7,8,[8,0],[5,6]
            nodes_size = 5;
            TF d_8_0 = di_8[ 0 ] / ( di_0[ 0 ] - di_8[ 0 ] );
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            px_1[ 0 ] = px_7[ 0 ];
            py_1[ 0 ] = py_7[ 0 ];
            px_2[ 0 ] = px_8[ 0 ];
            py_2[ 0 ] = py_8[ 0 ];
            px_3[ 0 ] = px_8[ 0 ] + d_8_0 * ( px_8[ 0 ] - px_0[ 0 ] );
            py_3[ 0 ] = py_8[ 0 ] + d_8_0 * ( py_8[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_4[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_4[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = pi_7[ 0 ];
            pi_2[ 0 ] = pi_8[ 0 ];
            pi_3[ 0 ] = cut_i[ num_cut ];
            pi_4[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_211: {
            // out: 1,1,1,1,1,1,0,0,1 cut: 7,[7,8],[5,6],6
            nodes_size = 4;
            TF d_8_7 = di_8[ 0 ] / ( di_7[ 0 ] - di_8[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_0[ 0 ] = px_7[ 0 ];
            py_0[ 0 ] = py_7[ 0 ];
            px_1[ 0 ] = px_8[ 0 ] + d_8_7 * ( px_8[ 0 ] - px_7[ 0 ] );
            py_1[ 0 ] = py_8[ 0 ] + d_8_7 * ( py_8[ 0 ] - py_7[ 0 ] );
            px_2[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_2[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            px_3[ 0 ] = px_6[ 0 ];
            py_3[ 0 ] = py_6[ 0 ];
            pi_0[ 0 ] = pi_7[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_6[ 0 ];
            continue;
        }
        
        case_147: {
            // out: 1,1,1,1,1,1,0,1 cut: 6,[6,7],[5,6]
            nodes_size = 3;
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            TF d_6_5 = di_6[ 0 ] / ( di_5[ 0 ] - di_6[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_1[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_1[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            px_2[ 0 ] = px_6[ 0 ] + d_6_5 * ( px_6[ 0 ] - px_5[ 0 ] );
            py_2[ 0 ] = py_6[ 0 ] + d_6_5 * ( py_6[ 0 ] - py_5[ 0 ] );
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_219: {
            // out: 1,1,1,1,1,1,0,1,1 cut: 6,[6,7],[5,6]
            nodes_size = 3;
            TF d_7_6 = di_7[ 0 ] / ( di_6[ 0 ] - di_7[ 0 ] );
            TF d_5_6 = di_5[ 0 ] / ( di_6[ 0 ] - di_5[ 0 ] );
            px_0[ 0 ] = px_6[ 0 ];
            py_0[ 0 ] = py_6[ 0 ];
            px_1[ 0 ] = px_7[ 0 ] + d_7_6 * ( px_7[ 0 ] - px_6[ 0 ] );
            py_1[ 0 ] = py_7[ 0 ] + d_7_6 * ( py_7[ 0 ] - py_6[ 0 ] );
            px_2[ 0 ] = px_5[ 0 ] + d_5_6 * ( px_5[ 0 ] - px_6[ 0 ] );
            py_2[ 0 ] = py_5[ 0 ] + d_5_6 * ( py_5[ 0 ] - py_6[ 0 ] );
            pi_0[ 0 ] = pi_6[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_140: {
            // out: 1,1,1,1,1,1,1,0 cut: 7,[7,0],[6,7]
            nodes_size = 3;
            TF d_0_7 = di_0[ 0 ] / ( di_7[ 0 ] - di_0[ 0 ] );
            TF d_6_7 = di_6[ 0 ] / ( di_7[ 0 ] - di_6[ 0 ] );
            px_1[ 0 ] = px_0[ 0 ] + d_0_7 * ( px_0[ 0 ] - px_7[ 0 ] );
            py_1[ 0 ] = py_0[ 0 ] + d_0_7 * ( py_0[ 0 ] - py_7[ 0 ] );
            px_0[ 0 ] = px_7[ 0 ];
            py_0[ 0 ] = py_7[ 0 ];
            px_2[ 0 ] = px_6[ 0 ] + d_6_7 * ( px_6[ 0 ] - px_7[ 0 ] );
            py_2[ 0 ] = py_6[ 0 ] + d_6_7 * ( py_6[ 0 ] - py_7[ 0 ] );
            pi_0[ 0 ] = pi_7[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_196: {
            // out: 1,1,1,1,1,1,1,0,0 cut: 8,[8,0],[6,7],7
            nodes_size = 4;
            TF d_8_0 = di_8[ 0 ] / ( di_0[ 0 ] - di_8[ 0 ] );
            TF d_7_6 = di_7[ 0 ] / ( di_6[ 0 ] - di_7[ 0 ] );
            px_1[ 0 ] = px_8[ 0 ] + d_8_0 * ( px_8[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_8[ 0 ] + d_8_0 * ( py_8[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = px_8[ 0 ];
            py_0[ 0 ] = py_8[ 0 ];
            px_2[ 0 ] = px_7[ 0 ] + d_7_6 * ( px_7[ 0 ] - px_6[ 0 ] );
            py_2[ 0 ] = py_7[ 0 ] + d_7_6 * ( py_7[ 0 ] - py_6[ 0 ] );
            px_3[ 0 ] = px_7[ 0 ];
            py_3[ 0 ] = py_7[ 0 ];
            pi_0[ 0 ] = pi_8[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            pi_3[ 0 ] = pi_7[ 0 ];
            continue;
        }
        
        case_212: {
            // out: 1,1,1,1,1,1,1,0,1 cut: 7,[7,8],[6,7]
            nodes_size = 3;
            TF d_7_8 = di_7[ 0 ] / ( di_8[ 0 ] - di_7[ 0 ] );
            TF d_7_6 = di_7[ 0 ] / ( di_6[ 0 ] - di_7[ 0 ] );
            px_0[ 0 ] = px_7[ 0 ];
            py_0[ 0 ] = py_7[ 0 ];
            px_1[ 0 ] = px_7[ 0 ] + d_7_8 * ( px_7[ 0 ] - px_8[ 0 ] );
            py_1[ 0 ] = py_7[ 0 ] + d_7_8 * ( py_7[ 0 ] - py_8[ 0 ] );
            px_2[ 0 ] = px_7[ 0 ] + d_7_6 * ( px_7[ 0 ] - px_6[ 0 ] );
            py_2[ 0 ] = py_7[ 0 ] + d_7_6 * ( py_7[ 0 ] - py_6[ 0 ] );
            pi_0[ 0 ] = pi_7[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
        
        case_204: {
            // out: 1,1,1,1,1,1,1,1,0 cut: 8,[8,0],[7,8]
            nodes_size = 3;
            TF d_8_0 = di_8[ 0 ] / ( di_0[ 0 ] - di_8[ 0 ] );
            TF d_8_7 = di_8[ 0 ] / ( di_7[ 0 ] - di_8[ 0 ] );
            px_1[ 0 ] = px_8[ 0 ] + d_8_0 * ( px_8[ 0 ] - px_0[ 0 ] );
            py_1[ 0 ] = py_8[ 0 ] + d_8_0 * ( py_8[ 0 ] - py_0[ 0 ] );
            px_0[ 0 ] = px_8[ 0 ];
            py_0[ 0 ] = py_8[ 0 ];
            px_2[ 0 ] = px_8[ 0 ] + d_8_7 * ( px_8[ 0 ] - px_7[ 0 ] );
            py_2[ 0 ] = py_8[ 0 ] + d_8_7 * ( py_8[ 0 ] - py_7[ 0 ] );
            pi_0[ 0 ] = pi_8[ 0 ];
            pi_1[ 0 ] = cut_i[ num_cut ];
            pi_2[ 0 ] = cut_i[ num_cut ];
            continue;
        }
    }
}
