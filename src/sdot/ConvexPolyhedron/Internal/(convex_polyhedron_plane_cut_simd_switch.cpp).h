#include "../ConvexPolyhedron2D.h"

namespace sdot {

template<class Pc> template<int flags,class T,int s>
void ConvexPolyhedron2D<Pc>::plane_cut_simd_switch( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags>, S<T>, N<s> ) {
    for( std::size_t i = 0; i < nb_cuts; ++i )
        plane_cut_gen( cut_dir[ 0 ][ i ], cut_dir[ 1 ][ i ], cut_ps[ i ], cut_id[ i ], N<flags>() );
}

template<class Pc> template<int flags>
void ConvexPolyhedron2D<Pc>::plane_cut_simd_switch( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags>, S<double>, N<8> ) {
    #ifdef __AVX512F__
    // outsize list
    TF *x = &nodes->x;
    TF *y = &nodes->y;
    CI *c = &nodes->cut_id.val;
    __m512d px_0 = _mm512_load_pd( x + 0 );
    __m512d py_0 = _mm512_load_pd( y + 0 );
    __m512i pc_0 = _mm512_load_epi64( c + 0 );
    for( std::size_t num_cut = 0; num_cut < nb_cuts; ++num_cut ) {
        __m512d rd = _mm512_set1_pd( cut_ps[ num_cut ] );
        __m512d nx = _mm512_set1_pd( cut_dir[ 0 ][ num_cut ] );
        __m512d ny = _mm512_set1_pd( cut_dir[ 1 ][ num_cut ] );
        __m512d bi_0 = _mm512_add_pd( _mm512_mul_pd( px_0, nx ), _mm512_mul_pd( py_0, ny ) );
        std::uint8_t outside_0 = _mm512_cmp_pd_mask( bi_0, rd, _CMP_GT_OQ );
        __m512d di_0 = _mm512_sub_pd( bi_0, rd );

        switch( 256 * size + 1 * outside_0 ) {
        case 769:
        case 777:
        case 785:
        case 793:
        case 801:
        case 809:
        case 817:
        case 825:
        case 833:
        case 841:
        case 849:
        case 857:
        case 865:
        case 873:
        case 881:
        case 889:
        case 897:
        case 905:
        case 913:
        case 921:
        case 929:
        case 937:
        case 945:
        case 953:
        case 961:
        case 969:
        case 977:
        case 985:
        case 993:
        case 1001:
        case 1009:
        case 1017: {
            // size=3 outside=0000000000000000000000000000000000000000000000000000000000000001 mod=[ 0, 1 ],1,2,[ 0, 2 ]
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9020108ul ) );
            __m128d d_i0 = _mm_set1_pd( d_0 );
            __m128d x_i0 = _mm_set1_pd( x_0 );
            __m128d y_i0 = _mm_set1_pd( y_0 );
            __m128d d_i1 = _mm_set_pd( d_2, d_1 );
            __m128d x_i1 = _mm_set_pd( x_2, x_1 );
            __m128d y_i1 = _mm_set_pd( y_2, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 770:
        case 778:
        case 786:
        case 794:
        case 802:
        case 810:
        case 818:
        case 826:
        case 834:
        case 842:
        case 850:
        case 858:
        case 866:
        case 874:
        case 882:
        case 890:
        case 898:
        case 906:
        case 914:
        case 922:
        case 930:
        case 938:
        case 946:
        case 954:
        case 962:
        case 970:
        case 978:
        case 986:
        case 994:
        case 1002:
        case 1010:
        case 1018: {
            // size=3 outside=0000000000000000000000000000000000000000000000000000000000000010 mod=0,[ 1, 0 ],[ 1, 2 ],2
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x2090800ul ) );
            __m128d d_i0 = _mm_set1_pd( d_1 );
            __m128d x_i0 = _mm_set1_pd( x_1 );
            __m128d y_i0 = _mm_set1_pd( y_1 );
            __m128d d_i1 = _mm_set_pd( d_2, d_0 );
            __m128d x_i1 = _mm_set_pd( x_2, x_0 );
            __m128d y_i1 = _mm_set_pd( y_2, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 771:
        case 779:
        case 787:
        case 795:
        case 803:
        case 811:
        case 819:
        case 827:
        case 835:
        case 843:
        case 851:
        case 859:
        case 867:
        case 875:
        case 883:
        case 891:
        case 899:
        case 907:
        case 915:
        case 923:
        case 931:
        case 939:
        case 947:
        case 955:
        case 963:
        case 971:
        case 979:
        case 987:
        case 995:
        case 1003:
        case 1011:
        case 1019: {
            // size=3 outside=0000000000000000000000000000000000000000000000000000000000000011 mod=[ 0, 2 ],[ 1, 2 ],2
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x20908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_0 );
            __m128d x_i0 = _mm_set_pd( x_1, x_0 );
            __m128d y_i0 = _mm_set_pd( y_1, y_0 );
            __m128d d_i1 = _mm_set1_pd( d_2 );
            __m128d x_i1 = _mm_set1_pd( x_2 );
            __m128d y_i1 = _mm_set1_pd( y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 772:
        case 780:
        case 788:
        case 796:
        case 804:
        case 812:
        case 820:
        case 828:
        case 836:
        case 844:
        case 852:
        case 860:
        case 868:
        case 876:
        case 884:
        case 892:
        case 900:
        case 908:
        case 916:
        case 924:
        case 932:
        case 940:
        case 948:
        case 956:
        case 964:
        case 972:
        case 980:
        case 988:
        case 996:
        case 1004:
        case 1012:
        case 1020: {
            // size=3 outside=0000000000000000000000000000000000000000000000000000000000000100 mod=0,1,[ 2, 1 ],[ 2, 0 ]
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9080100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_2 );
            __m128d x_i0 = _mm_set1_pd( x_2 );
            __m128d y_i0 = _mm_set1_pd( y_2 );
            __m128d d_i1 = _mm_set_pd( d_0, d_1 );
            __m128d x_i1 = _mm_set_pd( x_0, x_1 );
            __m128d y_i1 = _mm_set_pd( y_0, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 773:
        case 781:
        case 789:
        case 797:
        case 805:
        case 813:
        case 821:
        case 829:
        case 837:
        case 845:
        case 853:
        case 861:
        case 869:
        case 877:
        case 885:
        case 893:
        case 901:
        case 909:
        case 917:
        case 925:
        case 933:
        case 941:
        case 949:
        case 957:
        case 965:
        case 973:
        case 981:
        case 989:
        case 997:
        case 1005:
        case 1013:
        case 1021: {
            // size=3 outside=0000000000000000000000000000000000000000000000000000000000000101 mod=[ 0, 1 ],1,[ 2, 1 ]
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90108ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_0 );
            __m128d x_i0 = _mm_set_pd( x_2, x_0 );
            __m128d y_i0 = _mm_set_pd( y_2, y_0 );
            __m128d d_i1 = _mm_set1_pd( d_1 );
            __m128d x_i1 = _mm_set1_pd( x_1 );
            __m128d y_i1 = _mm_set1_pd( y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 774:
        case 782:
        case 790:
        case 798:
        case 806:
        case 814:
        case 822:
        case 830:
        case 838:
        case 846:
        case 854:
        case 862:
        case 870:
        case 878:
        case 886:
        case 894:
        case 902:
        case 910:
        case 918:
        case 926:
        case 934:
        case 942:
        case 950:
        case 958:
        case 966:
        case 974:
        case 982:
        case 990:
        case 998:
        case 1006:
        case 1014:
        case 1022: {
            // size=3 outside=0000000000000000000000000000000000000000000000000000000000000110 mod=0,[ 1, 0 ],[ 2, 0 ]
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90800ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_1 );
            __m128d x_i0 = _mm_set_pd( x_2, x_1 );
            __m128d y_i0 = _mm_set_pd( y_2, y_1 );
            __m128d d_i1 = _mm_set1_pd( d_0 );
            __m128d x_i1 = _mm_set1_pd( x_0 );
            __m128d y_i1 = _mm_set1_pd( y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1025:
        case 1041:
        case 1057:
        case 1073:
        case 1089:
        case 1105:
        case 1121:
        case 1137:
        case 1153:
        case 1169:
        case 1185:
        case 1201:
        case 1217:
        case 1233:
        case 1249:
        case 1265: {
            // size=4 outside=0000000000000000000000000000000000000000000000000000000000000001 mod=[ 0, 1 ],1,2,3,[ 0, 3 ]
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x903020108ul ) );
            __m128d d_i0 = _mm_set1_pd( d_0 );
            __m128d x_i0 = _mm_set1_pd( x_0 );
            __m128d y_i0 = _mm_set1_pd( y_0 );
            __m128d d_i1 = _mm_set_pd( d_3, d_1 );
            __m128d x_i1 = _mm_set_pd( x_3, x_1 );
            __m128d y_i1 = _mm_set_pd( y_3, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1026:
        case 1042:
        case 1058:
        case 1074:
        case 1090:
        case 1106:
        case 1122:
        case 1138:
        case 1154:
        case 1170:
        case 1186:
        case 1202:
        case 1218:
        case 1234:
        case 1250:
        case 1266: {
            // size=4 outside=0000000000000000000000000000000000000000000000000000000000000010 mod=[ 1, 0 ],[ 1, 2 ],2,3,0
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x3020908ul ) );
            __m128d d_i0 = _mm_set1_pd( d_1 );
            __m128d x_i0 = _mm_set1_pd( x_1 );
            __m128d y_i0 = _mm_set1_pd( y_1 );
            __m128d d_i1 = _mm_set_pd( d_2, d_0 );
            __m128d x_i1 = _mm_set_pd( x_2, x_0 );
            __m128d y_i1 = _mm_set_pd( y_2, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1027:
        case 1043:
        case 1059:
        case 1075:
        case 1091:
        case 1107:
        case 1123:
        case 1139:
        case 1155:
        case 1171:
        case 1187:
        case 1203:
        case 1219:
        case 1235:
        case 1251:
        case 1267: {
            // size=4 outside=0000000000000000000000000000000000000000000000000000000000000011 mod=[ 0, 3 ],[ 1, 2 ],2,3
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x3020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_0 );
            __m128d x_i0 = _mm_set_pd( x_1, x_0 );
            __m128d y_i0 = _mm_set_pd( y_1, y_0 );
            __m128d d_i1 = _mm_set_pd( d_2, d_3 );
            __m128d x_i1 = _mm_set_pd( x_2, x_3 );
            __m128d y_i1 = _mm_set_pd( y_2, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1028:
        case 1044:
        case 1060:
        case 1076:
        case 1092:
        case 1108:
        case 1124:
        case 1140:
        case 1156:
        case 1172:
        case 1188:
        case 1204:
        case 1220:
        case 1236:
        case 1252:
        case 1268: {
            // size=4 outside=0000000000000000000000000000000000000000000000000000000000000100 mod=0,1,[ 2, 1 ],[ 2, 3 ],3
            size = 5;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x309080100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_2 );
            __m128d x_i0 = _mm_set1_pd( x_2 );
            __m128d y_i0 = _mm_set1_pd( y_2 );
            __m128d d_i1 = _mm_set_pd( d_3, d_1 );
            __m128d x_i1 = _mm_set_pd( x_3, x_1 );
            __m128d y_i1 = _mm_set_pd( y_3, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1030:
        case 1046:
        case 1062:
        case 1078:
        case 1094:
        case 1110:
        case 1126:
        case 1142:
        case 1158:
        case 1174:
        case 1190:
        case 1206:
        case 1222:
        case 1238:
        case 1254:
        case 1270: {
            // size=4 outside=0000000000000000000000000000000000000000000000000000000000000110 mod=0,[ 1, 0 ],[ 2, 3 ],3
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x3090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_1 );
            __m128d x_i0 = _mm_set_pd( x_2, x_1 );
            __m128d y_i0 = _mm_set_pd( y_2, y_1 );
            __m128d d_i1 = _mm_set_pd( d_3, d_0 );
            __m128d x_i1 = _mm_set_pd( x_3, x_0 );
            __m128d y_i1 = _mm_set_pd( y_3, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1031:
        case 1047:
        case 1063:
        case 1079:
        case 1095:
        case 1111:
        case 1127:
        case 1143:
        case 1159:
        case 1175:
        case 1191:
        case 1207:
        case 1223:
        case 1239:
        case 1255:
        case 1271: {
            // size=4 outside=0000000000000000000000000000000000000000000000000000000000000111 mod=[ 0, 3 ],[ 2, 3 ],3
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x30908ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_0 );
            __m128d x_i0 = _mm_set_pd( x_2, x_0 );
            __m128d y_i0 = _mm_set_pd( y_2, y_0 );
            __m128d d_i1 = _mm_set1_pd( d_3 );
            __m128d x_i1 = _mm_set1_pd( x_3 );
            __m128d y_i1 = _mm_set1_pd( y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1032:
        case 1048:
        case 1064:
        case 1080:
        case 1096:
        case 1112:
        case 1128:
        case 1144:
        case 1160:
        case 1176:
        case 1192:
        case 1208:
        case 1224:
        case 1240:
        case 1256:
        case 1272: {
            // size=4 outside=0000000000000000000000000000000000000000000000000000000000001000 mod=0,1,2,[ 3, 2 ],[ 3, 0 ]
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x908020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_3 );
            __m128d x_i0 = _mm_set1_pd( x_3 );
            __m128d y_i0 = _mm_set1_pd( y_3 );
            __m128d d_i1 = _mm_set_pd( d_0, d_2 );
            __m128d x_i1 = _mm_set_pd( x_0, x_2 );
            __m128d y_i1 = _mm_set_pd( y_0, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1033:
        case 1049:
        case 1065:
        case 1081:
        case 1097:
        case 1113:
        case 1129:
        case 1145:
        case 1161:
        case 1177:
        case 1193:
        case 1209:
        case 1225:
        case 1241:
        case 1257:
        case 1273: {
            // size=4 outside=0000000000000000000000000000000000000000000000000000000000001001 mod=[ 0, 1 ],1,2,[ 3, 2 ]
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_0 );
            __m128d x_i0 = _mm_set_pd( x_3, x_0 );
            __m128d y_i0 = _mm_set_pd( y_3, y_0 );
            __m128d d_i1 = _mm_set_pd( d_2, d_1 );
            __m128d x_i1 = _mm_set_pd( x_2, x_1 );
            __m128d y_i1 = _mm_set_pd( y_2, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1035:
        case 1051:
        case 1067:
        case 1083:
        case 1099:
        case 1115:
        case 1131:
        case 1147:
        case 1163:
        case 1179:
        case 1195:
        case 1211:
        case 1227:
        case 1243:
        case 1259:
        case 1275: {
            // size=4 outside=0000000000000000000000000000000000000000000000000000000000001011 mod=[ 3, 2 ],[ 1, 2 ],2
            size = 3;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x20908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_3 );
            __m128d x_i0 = _mm_set_pd( x_1, x_3 );
            __m128d y_i0 = _mm_set_pd( y_1, y_3 );
            __m128d d_i1 = _mm_set1_pd( d_2 );
            __m128d x_i1 = _mm_set1_pd( x_2 );
            __m128d y_i1 = _mm_set1_pd( y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1036:
        case 1052:
        case 1068:
        case 1084:
        case 1100:
        case 1116:
        case 1132:
        case 1148:
        case 1164:
        case 1180:
        case 1196:
        case 1212:
        case 1228:
        case 1244:
        case 1260:
        case 1276: {
            // size=4 outside=0000000000000000000000000000000000000000000000000000000000001100 mod=0,1,[ 2, 1 ],[ 3, 0 ]
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_2 );
            __m128d x_i0 = _mm_set_pd( x_3, x_2 );
            __m128d y_i0 = _mm_set_pd( y_3, y_2 );
            __m128d d_i1 = _mm_set_pd( d_0, d_1 );
            __m128d x_i1 = _mm_set_pd( x_0, x_1 );
            __m128d y_i1 = _mm_set_pd( y_0, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1037:
        case 1053:
        case 1069:
        case 1085:
        case 1101:
        case 1117:
        case 1133:
        case 1149:
        case 1165:
        case 1181:
        case 1197:
        case 1213:
        case 1229:
        case 1245:
        case 1261:
        case 1277: {
            // size=4 outside=0000000000000000000000000000000000000000000000000000000000001101 mod=[ 0, 1 ],1,[ 2, 1 ]
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90108ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_0 );
            __m128d x_i0 = _mm_set_pd( x_2, x_0 );
            __m128d y_i0 = _mm_set_pd( y_2, y_0 );
            __m128d d_i1 = _mm_set1_pd( d_1 );
            __m128d x_i1 = _mm_set1_pd( x_1 );
            __m128d y_i1 = _mm_set1_pd( y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1038:
        case 1054:
        case 1070:
        case 1086:
        case 1102:
        case 1118:
        case 1134:
        case 1150:
        case 1166:
        case 1182:
        case 1198:
        case 1214:
        case 1230:
        case 1246:
        case 1262:
        case 1278: {
            // size=4 outside=0000000000000000000000000000000000000000000000000000000000001110 mod=0,[ 1, 0 ],[ 3, 0 ]
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90800ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_1 );
            __m128d x_i0 = _mm_set_pd( x_3, x_1 );
            __m128d y_i0 = _mm_set_pd( y_3, y_1 );
            __m128d d_i1 = _mm_set1_pd( d_0 );
            __m128d x_i1 = _mm_set1_pd( x_0 );
            __m128d y_i1 = _mm_set1_pd( y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1281:
        case 1313:
        case 1345:
        case 1377:
        case 1409:
        case 1441:
        case 1473:
        case 1505: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000000001 mod=[ 0, 1 ],1,2,3,4,[ 0, 4 ]
            size = 6;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90403020108ul ) );
            __m128d d_i0 = _mm_set1_pd( d_0 );
            __m128d x_i0 = _mm_set1_pd( x_0 );
            __m128d y_i0 = _mm_set1_pd( y_0 );
            __m128d d_i1 = _mm_set_pd( d_4, d_1 );
            __m128d x_i1 = _mm_set_pd( x_4, x_1 );
            __m128d y_i1 = _mm_set_pd( y_4, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1282:
        case 1314:
        case 1346:
        case 1378:
        case 1410:
        case 1442:
        case 1474:
        case 1506: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000000010 mod=[ 1, 0 ],[ 1, 2 ],2,3,4,0
            size = 6;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x403020908ul ) );
            __m128d d_i0 = _mm_set1_pd( d_1 );
            __m128d x_i0 = _mm_set1_pd( x_1 );
            __m128d y_i0 = _mm_set1_pd( y_1 );
            __m128d d_i1 = _mm_set_pd( d_2, d_0 );
            __m128d x_i1 = _mm_set_pd( x_2, x_0 );
            __m128d y_i1 = _mm_set_pd( y_2, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1283:
        case 1315:
        case 1347:
        case 1379:
        case 1411:
        case 1443:
        case 1475:
        case 1507: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000000011 mod=[ 0, 4 ],[ 1, 2 ],2,3,4
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x403020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_0 );
            __m128d x_i0 = _mm_set_pd( x_1, x_0 );
            __m128d y_i0 = _mm_set_pd( y_1, y_0 );
            __m128d d_i1 = _mm_set_pd( d_2, d_4 );
            __m128d x_i1 = _mm_set_pd( x_2, x_4 );
            __m128d y_i1 = _mm_set_pd( y_2, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1284:
        case 1316:
        case 1348:
        case 1380:
        case 1412:
        case 1444:
        case 1476:
        case 1508: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000000100 mod=0,1,[ 2, 1 ],[ 2, 3 ],3,4
            size = 6;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x40309080100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_2 );
            __m128d x_i0 = _mm_set1_pd( x_2 );
            __m128d y_i0 = _mm_set1_pd( y_2 );
            __m128d d_i1 = _mm_set_pd( d_3, d_1 );
            __m128d x_i1 = _mm_set_pd( x_3, x_1 );
            __m128d y_i1 = _mm_set_pd( y_3, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1286:
        case 1318:
        case 1350:
        case 1382:
        case 1414:
        case 1446:
        case 1478:
        case 1510: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000000110 mod=0,[ 1, 0 ],[ 2, 3 ],3,4
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x403090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_1 );
            __m128d x_i0 = _mm_set_pd( x_2, x_1 );
            __m128d y_i0 = _mm_set_pd( y_2, y_1 );
            __m128d d_i1 = _mm_set_pd( d_3, d_0 );
            __m128d x_i1 = _mm_set_pd( x_3, x_0 );
            __m128d y_i1 = _mm_set_pd( y_3, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1287:
        case 1319:
        case 1351:
        case 1383:
        case 1415:
        case 1447:
        case 1479:
        case 1511: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000000111 mod=4,[ 0, 4 ],[ 2, 3 ],3
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x3090804ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_0 );
            __m128d x_i0 = _mm_set_pd( x_2, x_0 );
            __m128d y_i0 = _mm_set_pd( y_2, y_0 );
            __m128d d_i1 = _mm_set_pd( d_3, d_4 );
            __m128d x_i1 = _mm_set_pd( x_3, x_4 );
            __m128d y_i1 = _mm_set_pd( y_3, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1288:
        case 1320:
        case 1352:
        case 1384:
        case 1416:
        case 1448:
        case 1480:
        case 1512: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000001000 mod=0,1,2,[ 3, 2 ],[ 3, 4 ],4
            size = 6;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x40908020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_3 );
            __m128d x_i0 = _mm_set1_pd( x_3 );
            __m128d y_i0 = _mm_set1_pd( y_3 );
            __m128d d_i1 = _mm_set_pd( d_4, d_2 );
            __m128d x_i1 = _mm_set_pd( x_4, x_2 );
            __m128d y_i1 = _mm_set_pd( y_4, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1292:
        case 1324:
        case 1356:
        case 1388:
        case 1420:
        case 1452:
        case 1484:
        case 1516: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000001100 mod=0,1,[ 2, 1 ],[ 3, 4 ],4
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x409080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_2 );
            __m128d x_i0 = _mm_set_pd( x_3, x_2 );
            __m128d y_i0 = _mm_set_pd( y_3, y_2 );
            __m128d d_i1 = _mm_set_pd( d_4, d_1 );
            __m128d x_i1 = _mm_set_pd( x_4, x_1 );
            __m128d y_i1 = _mm_set_pd( y_4, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1294:
        case 1326:
        case 1358:
        case 1390:
        case 1422:
        case 1454:
        case 1486:
        case 1518: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000001110 mod=0,[ 1, 0 ],[ 3, 4 ],4
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x4090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_1 );
            __m128d x_i0 = _mm_set_pd( x_3, x_1 );
            __m128d y_i0 = _mm_set_pd( y_3, y_1 );
            __m128d d_i1 = _mm_set_pd( d_4, d_0 );
            __m128d x_i1 = _mm_set_pd( x_4, x_0 );
            __m128d y_i1 = _mm_set_pd( y_4, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1295:
        case 1327:
        case 1359:
        case 1391:
        case 1423:
        case 1455:
        case 1487:
        case 1519: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000001111 mod=[ 0, 4 ],[ 3, 4 ],4
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x40908ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_0 );
            __m128d x_i0 = _mm_set_pd( x_3, x_0 );
            __m128d y_i0 = _mm_set_pd( y_3, y_0 );
            __m128d d_i1 = _mm_set1_pd( d_4 );
            __m128d x_i1 = _mm_set1_pd( x_4 );
            __m128d y_i1 = _mm_set1_pd( y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1296:
        case 1328:
        case 1360:
        case 1392:
        case 1424:
        case 1456:
        case 1488:
        case 1520: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000010000 mod=0,1,2,3,[ 4, 3 ],[ 4, 0 ]
            size = 6;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90803020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_4 );
            __m128d x_i0 = _mm_set1_pd( x_4 );
            __m128d y_i0 = _mm_set1_pd( y_4 );
            __m128d d_i1 = _mm_set_pd( d_0, d_3 );
            __m128d x_i1 = _mm_set_pd( x_0, x_3 );
            __m128d y_i1 = _mm_set_pd( y_0, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1297:
        case 1329:
        case 1361:
        case 1393:
        case 1425:
        case 1457:
        case 1489:
        case 1521: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000010001 mod=[ 0, 1 ],1,2,3,[ 4, 3 ]
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x903020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_0 );
            __m128d x_i0 = _mm_set_pd( x_4, x_0 );
            __m128d y_i0 = _mm_set_pd( y_4, y_0 );
            __m128d d_i1 = _mm_set_pd( d_3, d_1 );
            __m128d x_i1 = _mm_set_pd( x_3, x_1 );
            __m128d y_i1 = _mm_set_pd( y_3, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1299:
        case 1331:
        case 1363:
        case 1395:
        case 1427:
        case 1459:
        case 1491:
        case 1523: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000010011 mod=[ 4, 3 ],[ 1, 2 ],2,3
            size = 4;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x3020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_4 );
            __m128d x_i0 = _mm_set_pd( x_1, x_4 );
            __m128d y_i0 = _mm_set_pd( y_1, y_4 );
            __m128d d_i1 = _mm_set_pd( d_2, d_3 );
            __m128d x_i1 = _mm_set_pd( x_2, x_3 );
            __m128d y_i1 = _mm_set_pd( y_2, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1303:
        case 1335:
        case 1367:
        case 1399:
        case 1431:
        case 1463:
        case 1495:
        case 1527: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000010111 mod=3,[ 4, 3 ],[ 2, 3 ]
            size = 3;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90803ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_4 );
            __m128d x_i0 = _mm_set_pd( x_2, x_4 );
            __m128d y_i0 = _mm_set_pd( y_2, y_4 );
            __m128d d_i1 = _mm_set1_pd( d_3 );
            __m128d x_i1 = _mm_set1_pd( x_3 );
            __m128d y_i1 = _mm_set1_pd( y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1304:
        case 1336:
        case 1368:
        case 1400:
        case 1432:
        case 1464:
        case 1496:
        case 1528: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000011000 mod=0,1,2,[ 3, 2 ],[ 4, 0 ]
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x908020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_3 );
            __m128d x_i0 = _mm_set_pd( x_4, x_3 );
            __m128d y_i0 = _mm_set_pd( y_4, y_3 );
            __m128d d_i1 = _mm_set_pd( d_0, d_2 );
            __m128d x_i1 = _mm_set_pd( x_0, x_2 );
            __m128d y_i1 = _mm_set_pd( y_0, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1305:
        case 1337:
        case 1369:
        case 1401:
        case 1433:
        case 1465:
        case 1497:
        case 1529: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000011001 mod=[ 0, 1 ],1,2,[ 3, 2 ]
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_0 );
            __m128d x_i0 = _mm_set_pd( x_3, x_0 );
            __m128d y_i0 = _mm_set_pd( y_3, y_0 );
            __m128d d_i1 = _mm_set_pd( d_2, d_1 );
            __m128d x_i1 = _mm_set_pd( x_2, x_1 );
            __m128d y_i1 = _mm_set_pd( y_2, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1307:
        case 1339:
        case 1371:
        case 1403:
        case 1435:
        case 1467:
        case 1499:
        case 1531: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000011011 mod=[ 3, 2 ],[ 1, 2 ],2
            size = 3;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x20908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_3 );
            __m128d x_i0 = _mm_set_pd( x_1, x_3 );
            __m128d y_i0 = _mm_set_pd( y_1, y_3 );
            __m128d d_i1 = _mm_set1_pd( d_2 );
            __m128d x_i1 = _mm_set1_pd( x_2 );
            __m128d y_i1 = _mm_set1_pd( y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1308:
        case 1340:
        case 1372:
        case 1404:
        case 1436:
        case 1468:
        case 1500:
        case 1532: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000011100 mod=0,1,[ 2, 1 ],[ 4, 0 ]
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_2 );
            __m128d x_i0 = _mm_set_pd( x_4, x_2 );
            __m128d y_i0 = _mm_set_pd( y_4, y_2 );
            __m128d d_i1 = _mm_set_pd( d_0, d_1 );
            __m128d x_i1 = _mm_set_pd( x_0, x_1 );
            __m128d y_i1 = _mm_set_pd( y_0, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1309:
        case 1341:
        case 1373:
        case 1405:
        case 1437:
        case 1469:
        case 1501:
        case 1533: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000011101 mod=[ 0, 1 ],1,[ 2, 1 ]
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90108ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_0 );
            __m128d x_i0 = _mm_set_pd( x_2, x_0 );
            __m128d y_i0 = _mm_set_pd( y_2, y_0 );
            __m128d d_i1 = _mm_set1_pd( d_1 );
            __m128d x_i1 = _mm_set1_pd( x_1 );
            __m128d y_i1 = _mm_set1_pd( y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1310:
        case 1342:
        case 1374:
        case 1406:
        case 1438:
        case 1470:
        case 1502:
        case 1534: {
            // size=5 outside=0000000000000000000000000000000000000000000000000000000000011110 mod=0,[ 1, 0 ],[ 4, 0 ]
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90800ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_1 );
            __m128d x_i0 = _mm_set_pd( x_4, x_1 );
            __m128d y_i0 = _mm_set_pd( y_4, y_1 );
            __m128d d_i1 = _mm_set1_pd( d_0 );
            __m128d x_i1 = _mm_set1_pd( x_0 );
            __m128d y_i1 = _mm_set1_pd( y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1537:
        case 1601:
        case 1665:
        case 1729: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000000001 mod=[ 0, 1 ],1,2,3,4,5,[ 0, 5 ]
            size = 7;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9050403020108ul ) );
            __m128d d_i0 = _mm_set1_pd( d_0 );
            __m128d x_i0 = _mm_set1_pd( x_0 );
            __m128d y_i0 = _mm_set1_pd( y_0 );
            __m128d d_i1 = _mm_set_pd( d_5, d_1 );
            __m128d x_i1 = _mm_set_pd( x_5, x_1 );
            __m128d y_i1 = _mm_set_pd( y_5, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1538:
        case 1602:
        case 1666:
        case 1730: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000000010 mod=[ 1, 0 ],[ 1, 2 ],2,3,4,5,0
            size = 7;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50403020908ul ) );
            __m128d d_i0 = _mm_set1_pd( d_1 );
            __m128d x_i0 = _mm_set1_pd( x_1 );
            __m128d y_i0 = _mm_set1_pd( y_1 );
            __m128d d_i1 = _mm_set_pd( d_2, d_0 );
            __m128d x_i1 = _mm_set_pd( x_2, x_0 );
            __m128d y_i1 = _mm_set_pd( y_2, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1539:
        case 1603:
        case 1667:
        case 1731: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000000011 mod=[ 0, 5 ],[ 1, 2 ],2,3,4,5
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50403020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_0 );
            __m128d x_i0 = _mm_set_pd( x_1, x_0 );
            __m128d y_i0 = _mm_set_pd( y_1, y_0 );
            __m128d d_i1 = _mm_set_pd( d_2, d_5 );
            __m128d x_i1 = _mm_set_pd( x_2, x_5 );
            __m128d y_i1 = _mm_set_pd( y_2, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1540:
        case 1604:
        case 1668:
        case 1732: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000000100 mod=1,[ 2, 1 ],[ 2, 3 ],3,4,5,0
            size = 7;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50403090801ul ) );
            __m128d d_i0 = _mm_set1_pd( d_2 );
            __m128d x_i0 = _mm_set1_pd( x_2 );
            __m128d y_i0 = _mm_set1_pd( y_2 );
            __m128d d_i1 = _mm_set_pd( d_3, d_1 );
            __m128d x_i1 = _mm_set_pd( x_3, x_1 );
            __m128d y_i1 = _mm_set_pd( y_3, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1542:
        case 1606:
        case 1670:
        case 1734: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000000110 mod=0,[ 1, 0 ],[ 2, 3 ],3,4,5
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50403090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_1 );
            __m128d x_i0 = _mm_set_pd( x_2, x_1 );
            __m128d y_i0 = _mm_set_pd( y_2, y_1 );
            __m128d d_i1 = _mm_set_pd( d_3, d_0 );
            __m128d x_i1 = _mm_set_pd( x_3, x_0 );
            __m128d y_i1 = _mm_set_pd( y_3, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1543:
        case 1607:
        case 1671:
        case 1735: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000000111 mod=5,[ 0, 5 ],[ 2, 3 ],3,4
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x403090805ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_0 );
            __m128d x_i0 = _mm_set_pd( x_2, x_0 );
            __m128d y_i0 = _mm_set_pd( y_2, y_0 );
            __m128d d_i1 = _mm_set_pd( d_3, d_5 );
            __m128d x_i1 = _mm_set_pd( x_3, x_5 );
            __m128d y_i1 = _mm_set_pd( y_3, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1544:
        case 1608:
        case 1672:
        case 1736: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000001000 mod=0,1,2,[ 3, 2 ],[ 3, 4 ],4,5
            size = 7;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x5040908020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_3 );
            __m128d x_i0 = _mm_set1_pd( x_3 );
            __m128d y_i0 = _mm_set1_pd( y_3 );
            __m128d d_i1 = _mm_set_pd( d_4, d_2 );
            __m128d x_i1 = _mm_set_pd( x_4, x_2 );
            __m128d y_i1 = _mm_set_pd( y_4, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1548:
        case 1612:
        case 1676:
        case 1740: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000001100 mod=0,1,[ 2, 1 ],[ 3, 4 ],4,5
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50409080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_2 );
            __m128d x_i0 = _mm_set_pd( x_3, x_2 );
            __m128d y_i0 = _mm_set_pd( y_3, y_2 );
            __m128d d_i1 = _mm_set_pd( d_4, d_1 );
            __m128d x_i1 = _mm_set_pd( x_4, x_1 );
            __m128d y_i1 = _mm_set_pd( y_4, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1550:
        case 1614:
        case 1678:
        case 1742: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000001110 mod=0,[ 1, 0 ],[ 3, 4 ],4,5
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x504090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_1 );
            __m128d x_i0 = _mm_set_pd( x_3, x_1 );
            __m128d y_i0 = _mm_set_pd( y_3, y_1 );
            __m128d d_i1 = _mm_set_pd( d_4, d_0 );
            __m128d x_i1 = _mm_set_pd( x_4, x_0 );
            __m128d y_i1 = _mm_set_pd( y_4, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1551:
        case 1615:
        case 1679:
        case 1743: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000001111 mod=[ 0, 5 ],[ 3, 4 ],4,5
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x5040908ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_0 );
            __m128d x_i0 = _mm_set_pd( x_3, x_0 );
            __m128d y_i0 = _mm_set_pd( y_3, y_0 );
            __m128d d_i1 = _mm_set_pd( d_4, d_5 );
            __m128d x_i1 = _mm_set_pd( x_4, x_5 );
            __m128d y_i1 = _mm_set_pd( y_4, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1552:
        case 1616:
        case 1680:
        case 1744: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000010000 mod=0,1,2,3,[ 4, 3 ],[ 4, 5 ],5
            size = 7;
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x5090803020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_4 );
            __m128d x_i0 = _mm_set1_pd( x_4 );
            __m128d y_i0 = _mm_set1_pd( y_4 );
            __m128d d_i1 = _mm_set_pd( d_5, d_3 );
            __m128d x_i1 = _mm_set_pd( x_5, x_3 );
            __m128d y_i1 = _mm_set_pd( y_5, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1560:
        case 1624:
        case 1688:
        case 1752: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000011000 mod=0,1,2,[ 3, 2 ],[ 4, 5 ],5
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50908020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_3 );
            __m128d x_i0 = _mm_set_pd( x_4, x_3 );
            __m128d y_i0 = _mm_set_pd( y_4, y_3 );
            __m128d d_i1 = _mm_set_pd( d_5, d_2 );
            __m128d x_i1 = _mm_set_pd( x_5, x_2 );
            __m128d y_i1 = _mm_set_pd( y_5, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1564:
        case 1628:
        case 1692:
        case 1756: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000011100 mod=0,1,[ 2, 1 ],[ 4, 5 ],5
            size = 5;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x509080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_2 );
            __m128d x_i0 = _mm_set_pd( x_4, x_2 );
            __m128d y_i0 = _mm_set_pd( y_4, y_2 );
            __m128d d_i1 = _mm_set_pd( d_5, d_1 );
            __m128d x_i1 = _mm_set_pd( x_5, x_1 );
            __m128d y_i1 = _mm_set_pd( y_5, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1566:
        case 1630:
        case 1694:
        case 1758: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000011110 mod=0,[ 1, 0 ],[ 4, 5 ],5
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x5090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_1 );
            __m128d x_i0 = _mm_set_pd( x_4, x_1 );
            __m128d y_i0 = _mm_set_pd( y_4, y_1 );
            __m128d d_i1 = _mm_set_pd( d_5, d_0 );
            __m128d x_i1 = _mm_set_pd( x_5, x_0 );
            __m128d y_i1 = _mm_set_pd( y_5, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1567:
        case 1631:
        case 1695:
        case 1759: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000011111 mod=[ 0, 5 ],[ 4, 5 ],5
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50908ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_0 );
            __m128d x_i0 = _mm_set_pd( x_4, x_0 );
            __m128d y_i0 = _mm_set_pd( y_4, y_0 );
            __m128d d_i1 = _mm_set1_pd( d_5 );
            __m128d x_i1 = _mm_set1_pd( x_5 );
            __m128d y_i1 = _mm_set1_pd( y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1568:
        case 1632:
        case 1696:
        case 1760: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000100000 mod=0,1,2,3,4,[ 5, 4 ],[ 5, 0 ]
            size = 7;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9080403020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_5 );
            __m128d x_i0 = _mm_set1_pd( x_5 );
            __m128d y_i0 = _mm_set1_pd( y_5 );
            __m128d d_i1 = _mm_set_pd( d_0, d_4 );
            __m128d x_i1 = _mm_set_pd( x_0, x_4 );
            __m128d y_i1 = _mm_set_pd( y_0, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1569:
        case 1633:
        case 1697:
        case 1761: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000100001 mod=[ 0, 1 ],1,2,3,4,[ 5, 4 ]
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90403020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_0 );
            __m128d x_i0 = _mm_set_pd( x_5, x_0 );
            __m128d y_i0 = _mm_set_pd( y_5, y_0 );
            __m128d d_i1 = _mm_set_pd( d_4, d_1 );
            __m128d x_i1 = _mm_set_pd( x_4, x_1 );
            __m128d y_i1 = _mm_set_pd( y_4, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1571:
        case 1635:
        case 1699:
        case 1763: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000100011 mod=[ 5, 4 ],[ 1, 2 ],2,3,4
            size = 5;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x403020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_5 );
            __m128d x_i0 = _mm_set_pd( x_1, x_5 );
            __m128d y_i0 = _mm_set_pd( y_1, y_5 );
            __m128d d_i1 = _mm_set_pd( d_2, d_4 );
            __m128d x_i1 = _mm_set_pd( x_2, x_4 );
            __m128d y_i1 = _mm_set_pd( y_2, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1575:
        case 1639:
        case 1703:
        case 1767: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000100111 mod=4,[ 5, 4 ],[ 2, 3 ],3
            size = 4;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x3090804ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_5 );
            __m128d x_i0 = _mm_set_pd( x_2, x_5 );
            __m128d y_i0 = _mm_set_pd( y_2, y_5 );
            __m128d d_i1 = _mm_set_pd( d_3, d_4 );
            __m128d x_i1 = _mm_set_pd( x_3, x_4 );
            __m128d y_i1 = _mm_set_pd( y_3, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1583:
        case 1647:
        case 1711:
        case 1775: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000101111 mod=[ 3, 4 ],4,[ 5, 4 ]
            size = 3;
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90408ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_3 );
            __m128d x_i0 = _mm_set_pd( x_5, x_3 );
            __m128d y_i0 = _mm_set_pd( y_5, y_3 );
            __m128d d_i1 = _mm_set1_pd( d_4 );
            __m128d x_i1 = _mm_set1_pd( x_4 );
            __m128d y_i1 = _mm_set1_pd( y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1584:
        case 1648:
        case 1712:
        case 1776: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000110000 mod=0,1,2,3,[ 4, 3 ],[ 5, 0 ]
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90803020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_4 );
            __m128d x_i0 = _mm_set_pd( x_5, x_4 );
            __m128d y_i0 = _mm_set_pd( y_5, y_4 );
            __m128d d_i1 = _mm_set_pd( d_0, d_3 );
            __m128d x_i1 = _mm_set_pd( x_0, x_3 );
            __m128d y_i1 = _mm_set_pd( y_0, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1585:
        case 1649:
        case 1713:
        case 1777: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000110001 mod=[ 0, 1 ],1,2,3,[ 4, 3 ]
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x903020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_0 );
            __m128d x_i0 = _mm_set_pd( x_4, x_0 );
            __m128d y_i0 = _mm_set_pd( y_4, y_0 );
            __m128d d_i1 = _mm_set_pd( d_3, d_1 );
            __m128d x_i1 = _mm_set_pd( x_3, x_1 );
            __m128d y_i1 = _mm_set_pd( y_3, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1587:
        case 1651:
        case 1715:
        case 1779: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000110011 mod=[ 4, 3 ],[ 1, 2 ],2,3
            size = 4;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x3020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_4 );
            __m128d x_i0 = _mm_set_pd( x_1, x_4 );
            __m128d y_i0 = _mm_set_pd( y_1, y_4 );
            __m128d d_i1 = _mm_set_pd( d_2, d_3 );
            __m128d x_i1 = _mm_set_pd( x_2, x_3 );
            __m128d y_i1 = _mm_set_pd( y_2, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1591:
        case 1655:
        case 1719:
        case 1783: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000110111 mod=3,[ 4, 3 ],[ 2, 3 ]
            size = 3;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90803ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_4 );
            __m128d x_i0 = _mm_set_pd( x_2, x_4 );
            __m128d y_i0 = _mm_set_pd( y_2, y_4 );
            __m128d d_i1 = _mm_set1_pd( d_3 );
            __m128d x_i1 = _mm_set1_pd( x_3 );
            __m128d y_i1 = _mm_set1_pd( y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1592:
        case 1656:
        case 1720:
        case 1784: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000111000 mod=0,1,2,[ 3, 2 ],[ 5, 0 ]
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x908020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_3 );
            __m128d x_i0 = _mm_set_pd( x_5, x_3 );
            __m128d y_i0 = _mm_set_pd( y_5, y_3 );
            __m128d d_i1 = _mm_set_pd( d_0, d_2 );
            __m128d x_i1 = _mm_set_pd( x_0, x_2 );
            __m128d y_i1 = _mm_set_pd( y_0, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1593:
        case 1657:
        case 1721:
        case 1785: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000111001 mod=[ 0, 1 ],1,2,[ 3, 2 ]
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_0 );
            __m128d x_i0 = _mm_set_pd( x_3, x_0 );
            __m128d y_i0 = _mm_set_pd( y_3, y_0 );
            __m128d d_i1 = _mm_set_pd( d_2, d_1 );
            __m128d x_i1 = _mm_set_pd( x_2, x_1 );
            __m128d y_i1 = _mm_set_pd( y_2, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1595:
        case 1659:
        case 1723:
        case 1787: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000111011 mod=[ 3, 2 ],[ 1, 2 ],2
            size = 3;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x20908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_3 );
            __m128d x_i0 = _mm_set_pd( x_1, x_3 );
            __m128d y_i0 = _mm_set_pd( y_1, y_3 );
            __m128d d_i1 = _mm_set1_pd( d_2 );
            __m128d x_i1 = _mm_set1_pd( x_2 );
            __m128d y_i1 = _mm_set1_pd( y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1596:
        case 1660:
        case 1724:
        case 1788: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000111100 mod=0,1,[ 2, 1 ],[ 5, 0 ]
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_2 );
            __m128d x_i0 = _mm_set_pd( x_5, x_2 );
            __m128d y_i0 = _mm_set_pd( y_5, y_2 );
            __m128d d_i1 = _mm_set_pd( d_0, d_1 );
            __m128d x_i1 = _mm_set_pd( x_0, x_1 );
            __m128d y_i1 = _mm_set_pd( y_0, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1597:
        case 1661:
        case 1725:
        case 1789: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000111101 mod=[ 0, 1 ],1,[ 2, 1 ]
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90108ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_0 );
            __m128d x_i0 = _mm_set_pd( x_2, x_0 );
            __m128d y_i0 = _mm_set_pd( y_2, y_0 );
            __m128d d_i1 = _mm_set1_pd( d_1 );
            __m128d x_i1 = _mm_set1_pd( x_1 );
            __m128d y_i1 = _mm_set1_pd( y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1598:
        case 1662:
        case 1726:
        case 1790: {
            // size=6 outside=0000000000000000000000000000000000000000000000000000000000111110 mod=0,[ 1, 0 ],[ 5, 0 ]
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90800ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_1 );
            __m128d x_i0 = _mm_set_pd( x_5, x_1 );
            __m128d y_i0 = _mm_set_pd( y_5, y_1 );
            __m128d d_i1 = _mm_set1_pd( d_0 );
            __m128d x_i1 = _mm_set1_pd( x_0 );
            __m128d y_i1 = _mm_set1_pd( y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1793:
        case 1921: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000000001 mod=[ 0, 1 ],1,2,3,4,5,6,[ 0, 6 ]
            size = 8;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x906050403020108ul ) );
            __m128d d_i0 = _mm_set1_pd( d_0 );
            __m128d x_i0 = _mm_set1_pd( x_0 );
            __m128d y_i0 = _mm_set1_pd( y_0 );
            __m128d d_i1 = _mm_set_pd( d_6, d_1 );
            __m128d x_i1 = _mm_set_pd( x_6, x_1 );
            __m128d y_i1 = _mm_set_pd( y_6, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1794:
        case 1922: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000000010 mod=[ 1, 0 ],[ 1, 2 ],2,3,4,5,6,0
            size = 8;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x6050403020908ul ) );
            __m128d d_i0 = _mm_set1_pd( d_1 );
            __m128d x_i0 = _mm_set1_pd( x_1 );
            __m128d y_i0 = _mm_set1_pd( y_1 );
            __m128d d_i1 = _mm_set_pd( d_2, d_0 );
            __m128d x_i1 = _mm_set_pd( x_2, x_0 );
            __m128d y_i1 = _mm_set_pd( y_2, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1795:
        case 1923: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000000011 mod=[ 0, 6 ],[ 1, 2 ],2,3,4,5,6
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x6050403020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_0 );
            __m128d x_i0 = _mm_set_pd( x_1, x_0 );
            __m128d y_i0 = _mm_set_pd( y_1, y_0 );
            __m128d d_i1 = _mm_set_pd( d_2, d_6 );
            __m128d x_i1 = _mm_set_pd( x_2, x_6 );
            __m128d y_i1 = _mm_set_pd( y_2, y_6 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1796:
        case 1924: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000000100 mod=1,[ 2, 1 ],[ 2, 3 ],3,4,5,6,0
            size = 8;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x6050403090801ul ) );
            __m128d d_i0 = _mm_set1_pd( d_2 );
            __m128d x_i0 = _mm_set1_pd( x_2 );
            __m128d y_i0 = _mm_set1_pd( y_2 );
            __m128d d_i1 = _mm_set_pd( d_3, d_1 );
            __m128d x_i1 = _mm_set_pd( x_3, x_1 );
            __m128d y_i1 = _mm_set_pd( y_3, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1798:
        case 1926: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000000110 mod=0,[ 1, 0 ],[ 2, 3 ],3,4,5,6
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x6050403090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_1 );
            __m128d x_i0 = _mm_set_pd( x_2, x_1 );
            __m128d y_i0 = _mm_set_pd( y_2, y_1 );
            __m128d d_i1 = _mm_set_pd( d_3, d_0 );
            __m128d x_i1 = _mm_set_pd( x_3, x_0 );
            __m128d y_i1 = _mm_set_pd( y_3, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1799:
        case 1927: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000000111 mod=6,[ 0, 6 ],[ 2, 3 ],3,4,5
            size = 6;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50403090806ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_0 );
            __m128d x_i0 = _mm_set_pd( x_2, x_0 );
            __m128d y_i0 = _mm_set_pd( y_2, y_0 );
            __m128d d_i1 = _mm_set_pd( d_3, d_6 );
            __m128d x_i1 = _mm_set_pd( x_3, x_6 );
            __m128d y_i1 = _mm_set_pd( y_3, y_6 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1800:
        case 1928: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000001000 mod=0,1,2,[ 3, 2 ],[ 3, 4 ],4,5,6
            size = 8;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x605040908020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_3 );
            __m128d x_i0 = _mm_set1_pd( x_3 );
            __m128d y_i0 = _mm_set1_pd( y_3 );
            __m128d d_i1 = _mm_set_pd( d_4, d_2 );
            __m128d x_i1 = _mm_set_pd( x_4, x_2 );
            __m128d y_i1 = _mm_set_pd( y_4, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1804:
        case 1932: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000001100 mod=0,1,[ 2, 1 ],[ 3, 4 ],4,5,6
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x6050409080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_2 );
            __m128d x_i0 = _mm_set_pd( x_3, x_2 );
            __m128d y_i0 = _mm_set_pd( y_3, y_2 );
            __m128d d_i1 = _mm_set_pd( d_4, d_1 );
            __m128d x_i1 = _mm_set_pd( x_4, x_1 );
            __m128d y_i1 = _mm_set_pd( y_4, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1806:
        case 1934: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000001110 mod=6,0,[ 1, 0 ],[ 3, 4 ],4,5
            size = 6;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50409080006ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_1 );
            __m128d x_i0 = _mm_set_pd( x_3, x_1 );
            __m128d y_i0 = _mm_set_pd( y_3, y_1 );
            __m128d d_i1 = _mm_set_pd( d_4, d_0 );
            __m128d x_i1 = _mm_set_pd( x_4, x_0 );
            __m128d y_i1 = _mm_set_pd( y_4, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1807:
        case 1935: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000001111 mod=5,6,[ 0, 6 ],[ 3, 4 ],4
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x409080605ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_0 );
            __m128d x_i0 = _mm_set_pd( x_3, x_0 );
            __m128d y_i0 = _mm_set_pd( y_3, y_0 );
            __m128d d_i1 = _mm_set_pd( d_4, d_6 );
            __m128d x_i1 = _mm_set_pd( x_4, x_6 );
            __m128d y_i1 = _mm_set_pd( y_4, y_6 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1808:
        case 1936: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000010000 mod=0,1,2,3,[ 4, 3 ],[ 4, 5 ],5,6
            size = 8;
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x605090803020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_4 );
            __m128d x_i0 = _mm_set1_pd( x_4 );
            __m128d y_i0 = _mm_set1_pd( y_4 );
            __m128d d_i1 = _mm_set_pd( d_5, d_3 );
            __m128d x_i1 = _mm_set_pd( x_5, x_3 );
            __m128d y_i1 = _mm_set_pd( y_5, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1816:
        case 1944: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000011000 mod=0,1,2,[ 3, 2 ],[ 4, 5 ],5,6
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x6050908020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_3 );
            __m128d x_i0 = _mm_set_pd( x_4, x_3 );
            __m128d y_i0 = _mm_set_pd( y_4, y_3 );
            __m128d d_i1 = _mm_set_pd( d_5, d_2 );
            __m128d x_i1 = _mm_set_pd( x_5, x_2 );
            __m128d y_i1 = _mm_set_pd( y_5, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1820:
        case 1948: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000011100 mod=0,1,[ 2, 1 ],[ 4, 5 ],5,6
            size = 6;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x60509080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_2 );
            __m128d x_i0 = _mm_set_pd( x_4, x_2 );
            __m128d y_i0 = _mm_set_pd( y_4, y_2 );
            __m128d d_i1 = _mm_set_pd( d_5, d_1 );
            __m128d x_i1 = _mm_set_pd( x_5, x_1 );
            __m128d y_i1 = _mm_set_pd( y_5, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1822:
        case 1950: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000011110 mod=0,[ 1, 0 ],[ 4, 5 ],5,6
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x605090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_1 );
            __m128d x_i0 = _mm_set_pd( x_4, x_1 );
            __m128d y_i0 = _mm_set_pd( y_4, y_1 );
            __m128d d_i1 = _mm_set_pd( d_5, d_0 );
            __m128d x_i1 = _mm_set_pd( x_5, x_0 );
            __m128d y_i1 = _mm_set_pd( y_5, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1823:
        case 1951: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000011111 mod=[ 0, 6 ],[ 4, 5 ],5,6
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x6050908ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_0 );
            __m128d x_i0 = _mm_set_pd( x_4, x_0 );
            __m128d y_i0 = _mm_set_pd( y_4, y_0 );
            __m128d d_i1 = _mm_set_pd( d_5, d_6 );
            __m128d x_i1 = _mm_set_pd( x_5, x_6 );
            __m128d y_i1 = _mm_set_pd( y_5, y_6 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1824:
        case 1952: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000100000 mod=0,1,2,3,4,[ 5, 4 ],[ 5, 6 ],6
            size = 8;
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x609080403020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_5 );
            __m128d x_i0 = _mm_set1_pd( x_5 );
            __m128d y_i0 = _mm_set1_pd( y_5 );
            __m128d d_i1 = _mm_set_pd( d_6, d_4 );
            __m128d x_i1 = _mm_set_pd( x_6, x_4 );
            __m128d y_i1 = _mm_set_pd( y_6, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1840:
        case 1968: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000110000 mod=0,1,2,3,[ 4, 3 ],[ 5, 6 ],6
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x6090803020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_4 );
            __m128d x_i0 = _mm_set_pd( x_5, x_4 );
            __m128d y_i0 = _mm_set_pd( y_5, y_4 );
            __m128d d_i1 = _mm_set_pd( d_6, d_3 );
            __m128d x_i1 = _mm_set_pd( x_6, x_3 );
            __m128d y_i1 = _mm_set_pd( y_6, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1848:
        case 1976: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000111000 mod=0,1,2,[ 3, 2 ],[ 5, 6 ],6
            size = 6;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x60908020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_3 );
            __m128d x_i0 = _mm_set_pd( x_5, x_3 );
            __m128d y_i0 = _mm_set_pd( y_5, y_3 );
            __m128d d_i1 = _mm_set_pd( d_6, d_2 );
            __m128d x_i1 = _mm_set_pd( x_6, x_2 );
            __m128d y_i1 = _mm_set_pd( y_6, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1852:
        case 1980: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000111100 mod=0,1,[ 2, 1 ],[ 5, 6 ],6
            size = 5;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x609080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_2 );
            __m128d x_i0 = _mm_set_pd( x_5, x_2 );
            __m128d y_i0 = _mm_set_pd( y_5, y_2 );
            __m128d d_i1 = _mm_set_pd( d_6, d_1 );
            __m128d x_i1 = _mm_set_pd( x_6, x_1 );
            __m128d y_i1 = _mm_set_pd( y_6, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1854:
        case 1982: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000111110 mod=0,[ 1, 0 ],[ 5, 6 ],6
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x6090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_1 );
            __m128d x_i0 = _mm_set_pd( x_5, x_1 );
            __m128d y_i0 = _mm_set_pd( y_5, y_1 );
            __m128d d_i1 = _mm_set_pd( d_6, d_0 );
            __m128d x_i1 = _mm_set_pd( x_6, x_0 );
            __m128d y_i1 = _mm_set_pd( y_6, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1855:
        case 1983: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000000111111 mod=[ 0, 6 ],[ 5, 6 ],6
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x60908ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_0 );
            __m128d x_i0 = _mm_set_pd( x_5, x_0 );
            __m128d y_i0 = _mm_set_pd( y_5, y_0 );
            __m128d d_i1 = _mm_set1_pd( d_6 );
            __m128d x_i1 = _mm_set1_pd( x_6 );
            __m128d y_i1 = _mm_set1_pd( y_6 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1856:
        case 1984: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001000000 mod=0,1,2,3,4,5,[ 6, 5 ],[ 6, 0 ]
            size = 8;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x908050403020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_6 );
            __m128d x_i0 = _mm_set1_pd( x_6 );
            __m128d y_i0 = _mm_set1_pd( y_6 );
            __m128d d_i1 = _mm_set_pd( d_0, d_5 );
            __m128d x_i1 = _mm_set_pd( x_0, x_5 );
            __m128d y_i1 = _mm_set_pd( y_0, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1857:
        case 1985: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001000001 mod=[ 0, 1 ],1,2,3,4,5,[ 6, 5 ]
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9050403020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_0 );
            __m128d x_i0 = _mm_set_pd( x_6, x_0 );
            __m128d y_i0 = _mm_set_pd( y_6, y_0 );
            __m128d d_i1 = _mm_set_pd( d_5, d_1 );
            __m128d x_i1 = _mm_set_pd( x_5, x_1 );
            __m128d y_i1 = _mm_set_pd( y_5, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1859:
        case 1987: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001000011 mod=[ 6, 5 ],[ 1, 2 ],2,3,4,5
            size = 6;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50403020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_6 );
            __m128d x_i0 = _mm_set_pd( x_1, x_6 );
            __m128d y_i0 = _mm_set_pd( y_1, y_6 );
            __m128d d_i1 = _mm_set_pd( d_2, d_5 );
            __m128d x_i1 = _mm_set_pd( x_2, x_5 );
            __m128d y_i1 = _mm_set_pd( y_2, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1863:
        case 1991: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001000111 mod=5,[ 6, 5 ],[ 2, 3 ],3,4
            size = 5;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x403090805ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_6 );
            __m128d x_i0 = _mm_set_pd( x_2, x_6 );
            __m128d y_i0 = _mm_set_pd( y_2, y_6 );
            __m128d d_i1 = _mm_set_pd( d_3, d_5 );
            __m128d x_i1 = _mm_set_pd( x_3, x_5 );
            __m128d y_i1 = _mm_set_pd( y_3, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1871:
        case 1999: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001001111 mod=4,5,[ 6, 5 ],[ 3, 4 ]
            size = 4;
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9080504ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_6 );
            __m128d x_i0 = _mm_set_pd( x_3, x_6 );
            __m128d y_i0 = _mm_set_pd( y_3, y_6 );
            __m128d d_i1 = _mm_set_pd( d_4, d_5 );
            __m128d x_i1 = _mm_set_pd( x_4, x_5 );
            __m128d y_i1 = _mm_set_pd( y_4, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1887:
        case 2015: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001011111 mod=[ 4, 5 ],5,[ 6, 5 ]
            size = 3;
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90508ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_4 );
            __m128d x_i0 = _mm_set_pd( x_6, x_4 );
            __m128d y_i0 = _mm_set_pd( y_6, y_4 );
            __m128d d_i1 = _mm_set1_pd( d_5 );
            __m128d x_i1 = _mm_set1_pd( x_5 );
            __m128d y_i1 = _mm_set1_pd( y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1888:
        case 2016: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001100000 mod=0,1,2,3,4,[ 5, 4 ],[ 6, 0 ]
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9080403020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_5 );
            __m128d x_i0 = _mm_set_pd( x_6, x_5 );
            __m128d y_i0 = _mm_set_pd( y_6, y_5 );
            __m128d d_i1 = _mm_set_pd( d_0, d_4 );
            __m128d x_i1 = _mm_set_pd( x_0, x_4 );
            __m128d y_i1 = _mm_set_pd( y_0, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1889:
        case 2017: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001100001 mod=[ 0, 1 ],1,2,3,4,[ 5, 4 ]
            size = 6;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90403020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_0 );
            __m128d x_i0 = _mm_set_pd( x_5, x_0 );
            __m128d y_i0 = _mm_set_pd( y_5, y_0 );
            __m128d d_i1 = _mm_set_pd( d_4, d_1 );
            __m128d x_i1 = _mm_set_pd( x_4, x_1 );
            __m128d y_i1 = _mm_set_pd( y_4, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1891:
        case 2019: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001100011 mod=[ 5, 4 ],[ 1, 2 ],2,3,4
            size = 5;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x403020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_5 );
            __m128d x_i0 = _mm_set_pd( x_1, x_5 );
            __m128d y_i0 = _mm_set_pd( y_1, y_5 );
            __m128d d_i1 = _mm_set_pd( d_2, d_4 );
            __m128d x_i1 = _mm_set_pd( x_2, x_4 );
            __m128d y_i1 = _mm_set_pd( y_2, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1895:
        case 2023: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001100111 mod=4,[ 5, 4 ],[ 2, 3 ],3
            size = 4;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x3090804ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_5 );
            __m128d x_i0 = _mm_set_pd( x_2, x_5 );
            __m128d y_i0 = _mm_set_pd( y_2, y_5 );
            __m128d d_i1 = _mm_set_pd( d_3, d_4 );
            __m128d x_i1 = _mm_set_pd( x_3, x_4 );
            __m128d y_i1 = _mm_set_pd( y_3, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1903:
        case 2031: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001101111 mod=[ 3, 4 ],4,[ 5, 4 ]
            size = 3;
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90408ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_3 );
            __m128d x_i0 = _mm_set_pd( x_5, x_3 );
            __m128d y_i0 = _mm_set_pd( y_5, y_3 );
            __m128d d_i1 = _mm_set1_pd( d_4 );
            __m128d x_i1 = _mm_set1_pd( x_4 );
            __m128d y_i1 = _mm_set1_pd( y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1904:
        case 2032: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001110000 mod=0,1,2,3,[ 4, 3 ],[ 6, 0 ]
            size = 6;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90803020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_4 );
            __m128d x_i0 = _mm_set_pd( x_6, x_4 );
            __m128d y_i0 = _mm_set_pd( y_6, y_4 );
            __m128d d_i1 = _mm_set_pd( d_0, d_3 );
            __m128d x_i1 = _mm_set_pd( x_0, x_3 );
            __m128d y_i1 = _mm_set_pd( y_0, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1905:
        case 2033: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001110001 mod=[ 0, 1 ],1,2,3,[ 4, 3 ]
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x903020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_0 );
            __m128d x_i0 = _mm_set_pd( x_4, x_0 );
            __m128d y_i0 = _mm_set_pd( y_4, y_0 );
            __m128d d_i1 = _mm_set_pd( d_3, d_1 );
            __m128d x_i1 = _mm_set_pd( x_3, x_1 );
            __m128d y_i1 = _mm_set_pd( y_3, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1907:
        case 2035: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001110011 mod=[ 4, 3 ],[ 1, 2 ],2,3
            size = 4;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x3020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_4 );
            __m128d x_i0 = _mm_set_pd( x_1, x_4 );
            __m128d y_i0 = _mm_set_pd( y_1, y_4 );
            __m128d d_i1 = _mm_set_pd( d_2, d_3 );
            __m128d x_i1 = _mm_set_pd( x_2, x_3 );
            __m128d y_i1 = _mm_set_pd( y_2, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1911:
        case 2039: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001110111 mod=3,[ 4, 3 ],[ 2, 3 ]
            size = 3;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90803ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_4 );
            __m128d x_i0 = _mm_set_pd( x_2, x_4 );
            __m128d y_i0 = _mm_set_pd( y_2, y_4 );
            __m128d d_i1 = _mm_set1_pd( d_3 );
            __m128d x_i1 = _mm_set1_pd( x_3 );
            __m128d y_i1 = _mm_set1_pd( y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1912:
        case 2040: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001111000 mod=0,1,2,[ 3, 2 ],[ 6, 0 ]
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x908020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_3 );
            __m128d x_i0 = _mm_set_pd( x_6, x_3 );
            __m128d y_i0 = _mm_set_pd( y_6, y_3 );
            __m128d d_i1 = _mm_set_pd( d_0, d_2 );
            __m128d x_i1 = _mm_set_pd( x_0, x_2 );
            __m128d y_i1 = _mm_set_pd( y_0, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1913:
        case 2041: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001111001 mod=[ 0, 1 ],1,2,[ 3, 2 ]
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_0 );
            __m128d x_i0 = _mm_set_pd( x_3, x_0 );
            __m128d y_i0 = _mm_set_pd( y_3, y_0 );
            __m128d d_i1 = _mm_set_pd( d_2, d_1 );
            __m128d x_i1 = _mm_set_pd( x_2, x_1 );
            __m128d y_i1 = _mm_set_pd( y_2, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1915:
        case 2043: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001111011 mod=[ 3, 2 ],[ 1, 2 ],2
            size = 3;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x20908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_3 );
            __m128d x_i0 = _mm_set_pd( x_1, x_3 );
            __m128d y_i0 = _mm_set_pd( y_1, y_3 );
            __m128d d_i1 = _mm_set1_pd( d_2 );
            __m128d x_i1 = _mm_set1_pd( x_2 );
            __m128d y_i1 = _mm_set1_pd( y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1916:
        case 2044: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001111100 mod=0,1,[ 2, 1 ],[ 6, 0 ]
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_2 );
            __m128d x_i0 = _mm_set_pd( x_6, x_2 );
            __m128d y_i0 = _mm_set_pd( y_6, y_2 );
            __m128d d_i1 = _mm_set_pd( d_0, d_1 );
            __m128d x_i1 = _mm_set_pd( x_0, x_1 );
            __m128d y_i1 = _mm_set_pd( y_0, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1917:
        case 2045: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001111101 mod=[ 0, 1 ],1,[ 2, 1 ]
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90108ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_0 );
            __m128d x_i0 = _mm_set_pd( x_2, x_0 );
            __m128d y_i0 = _mm_set_pd( y_2, y_0 );
            __m128d d_i1 = _mm_set1_pd( d_1 );
            __m128d x_i1 = _mm_set1_pd( x_1 );
            __m128d y_i1 = _mm_set1_pd( y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 1918:
        case 2046: {
            // size=7 outside=0000000000000000000000000000000000000000000000000000000001111110 mod=0,[ 1, 0 ],[ 6, 0 ]
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90800ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_1 );
            __m128d x_i0 = _mm_set_pd( x_6, x_1 );
            __m128d y_i0 = _mm_set_pd( y_6, y_1 );
            __m128d d_i1 = _mm_set1_pd( d_0 );
            __m128d x_i1 = _mm_set1_pd( x_0 );
            __m128d y_i1 = _mm_set1_pd( y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2049: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000000001 mod=[ 0, 1 ],1,2,3,4,5,6,7,[ 0, 7 ]
            size = 9;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            TF m_0_7 = d_0 / ( d_7 - d_0 );
            x[ 8 ] = x_0 - m_0_7 * ( x_7 - x_0 );
            y[ 8 ] = y_0 - m_0_7 * ( y_7 - y_0 );
            c[ 8 ] = cut_id[ num_cut ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x706050403020108ul ) );
            TF m = d_0 / ( d_1 - d_0 ); // 1
            __m512d inter_x = _mm512_set1_pd( x_0 - m * ( x_1 - x_0 ) );
            __m512d inter_y = _mm512_set1_pd( y_0 - m * ( y_1 - y_0 ) );
            __m512i inter_c = _mm512_set1_epi64( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2050: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000000010 mod=[ 1, 0 ],[ 1, 2 ],2,3,4,5,6,7,0
            size = 9;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            x[ 8 ] = x_0;
            y[ 8 ] = y_0;
            c[ 8 ] = reinterpret_cast<const CI *>( &pc_0 )[ 0 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x706050403020908ul ) );
            __m128d d_i0 = _mm_set1_pd( d_1 );
            __m128d x_i0 = _mm_set1_pd( x_1 );
            __m128d y_i0 = _mm_set1_pd( y_1 );
            __m128d d_i1 = _mm_set_pd( d_2, d_0 );
            __m128d x_i1 = _mm_set_pd( x_2, x_0 );
            __m128d y_i1 = _mm_set_pd( y_2, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2051: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000000011 mod=[ 0, 7 ],[ 1, 2 ],2,3,4,5,6,7
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x706050403020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_0 );
            __m128d x_i0 = _mm_set_pd( x_1, x_0 );
            __m128d y_i0 = _mm_set_pd( y_1, y_0 );
            __m128d d_i1 = _mm_set_pd( d_2, d_7 );
            __m128d x_i1 = _mm_set_pd( x_2, x_7 );
            __m128d y_i1 = _mm_set_pd( y_2, y_7 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2052: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000000100 mod=1,[ 2, 1 ],[ 2, 3 ],3,4,5,6,7,0
            size = 9;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            x[ 8 ] = x_0;
            y[ 8 ] = y_0;
            c[ 8 ] = reinterpret_cast<const CI *>( &pc_0 )[ 0 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x706050403090801ul ) );
            __m128d d_i0 = _mm_set1_pd( d_2 );
            __m128d x_i0 = _mm_set1_pd( x_2 );
            __m128d y_i0 = _mm_set1_pd( y_2 );
            __m128d d_i1 = _mm_set_pd( d_3, d_1 );
            __m128d x_i1 = _mm_set_pd( x_3, x_1 );
            __m128d y_i1 = _mm_set_pd( y_3, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2054: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000000110 mod=0,[ 1, 0 ],[ 2, 3 ],3,4,5,6,7
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x706050403090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_1 );
            __m128d x_i0 = _mm_set_pd( x_2, x_1 );
            __m128d y_i0 = _mm_set_pd( y_2, y_1 );
            __m128d d_i1 = _mm_set_pd( d_3, d_0 );
            __m128d x_i1 = _mm_set_pd( x_3, x_0 );
            __m128d y_i1 = _mm_set_pd( y_3, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2055: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000000111 mod=7,[ 0, 7 ],[ 2, 3 ],3,4,5,6
            size = 7;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x6050403090807ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_0 );
            __m128d x_i0 = _mm_set_pd( x_2, x_0 );
            __m128d y_i0 = _mm_set_pd( y_2, y_0 );
            __m128d d_i1 = _mm_set_pd( d_3, d_7 );
            __m128d x_i1 = _mm_set_pd( x_3, x_7 );
            __m128d y_i1 = _mm_set_pd( y_3, y_7 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2056: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000001000 mod=1,2,[ 3, 2 ],[ 3, 4 ],4,5,6,7,0
            size = 9;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            x[ 8 ] = x_0;
            y[ 8 ] = y_0;
            c[ 8 ] = reinterpret_cast<const CI *>( &pc_0 )[ 0 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x706050409080201ul ) );
            __m128d d_i0 = _mm_set1_pd( d_3 );
            __m128d x_i0 = _mm_set1_pd( x_3 );
            __m128d y_i0 = _mm_set1_pd( y_3 );
            __m128d d_i1 = _mm_set_pd( d_4, d_2 );
            __m128d x_i1 = _mm_set_pd( x_4, x_2 );
            __m128d y_i1 = _mm_set_pd( y_4, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2060: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000001100 mod=0,1,[ 2, 1 ],[ 3, 4 ],4,5,6,7
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x706050409080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_2 );
            __m128d x_i0 = _mm_set_pd( x_3, x_2 );
            __m128d y_i0 = _mm_set_pd( y_3, y_2 );
            __m128d d_i1 = _mm_set_pd( d_4, d_1 );
            __m128d x_i1 = _mm_set_pd( x_4, x_1 );
            __m128d y_i1 = _mm_set_pd( y_4, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2062: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000001110 mod=7,0,[ 1, 0 ],[ 3, 4 ],4,5,6
            size = 7;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x6050409080007ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_1 );
            __m128d x_i0 = _mm_set_pd( x_3, x_1 );
            __m128d y_i0 = _mm_set_pd( y_3, y_1 );
            __m128d d_i1 = _mm_set_pd( d_4, d_0 );
            __m128d x_i1 = _mm_set_pd( x_4, x_0 );
            __m128d y_i1 = _mm_set_pd( y_4, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2063: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000001111 mod=6,7,[ 0, 7 ],[ 3, 4 ],4,5
            size = 6;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50409080706ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_0 );
            __m128d x_i0 = _mm_set_pd( x_3, x_0 );
            __m128d y_i0 = _mm_set_pd( y_3, y_0 );
            __m128d d_i1 = _mm_set_pd( d_4, d_7 );
            __m128d x_i1 = _mm_set_pd( x_4, x_7 );
            __m128d y_i1 = _mm_set_pd( y_4, y_7 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2064: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000010000 mod=0,1,2,3,[ 4, 3 ],[ 4, 5 ],5,6,7
            size = 9;
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            x[ 8 ] = x_7;
            y[ 8 ] = y_7;
            c[ 8 ] = reinterpret_cast<const CI *>( &pc_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x605090803020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_4 );
            __m128d x_i0 = _mm_set1_pd( x_4 );
            __m128d y_i0 = _mm_set1_pd( y_4 );
            __m128d d_i1 = _mm_set_pd( d_5, d_3 );
            __m128d x_i1 = _mm_set_pd( x_5, x_3 );
            __m128d y_i1 = _mm_set_pd( y_5, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2072: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000011000 mod=0,1,2,[ 3, 2 ],[ 4, 5 ],5,6,7
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x706050908020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_3 );
            __m128d x_i0 = _mm_set_pd( x_4, x_3 );
            __m128d y_i0 = _mm_set_pd( y_4, y_3 );
            __m128d d_i1 = _mm_set_pd( d_5, d_2 );
            __m128d x_i1 = _mm_set_pd( x_5, x_2 );
            __m128d y_i1 = _mm_set_pd( y_5, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2076: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000011100 mod=0,1,[ 2, 1 ],[ 4, 5 ],5,6,7
            size = 7;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x7060509080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_2 );
            __m128d x_i0 = _mm_set_pd( x_4, x_2 );
            __m128d y_i0 = _mm_set_pd( y_4, y_2 );
            __m128d d_i1 = _mm_set_pd( d_5, d_1 );
            __m128d x_i1 = _mm_set_pd( x_5, x_1 );
            __m128d y_i1 = _mm_set_pd( y_5, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2078: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000011110 mod=0,[ 1, 0 ],[ 4, 5 ],5,6,7
            size = 6;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x70605090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_1 );
            __m128d x_i0 = _mm_set_pd( x_4, x_1 );
            __m128d y_i0 = _mm_set_pd( y_4, y_1 );
            __m128d d_i1 = _mm_set_pd( d_5, d_0 );
            __m128d x_i1 = _mm_set_pd( x_5, x_0 );
            __m128d y_i1 = _mm_set_pd( y_5, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2079: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000011111 mod=[ 0, 7 ],[ 4, 5 ],5,6,7
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x706050908ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_0 );
            __m128d x_i0 = _mm_set_pd( x_4, x_0 );
            __m128d y_i0 = _mm_set_pd( y_4, y_0 );
            __m128d d_i1 = _mm_set_pd( d_5, d_7 );
            __m128d x_i1 = _mm_set_pd( x_5, x_7 );
            __m128d y_i1 = _mm_set_pd( y_5, y_7 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2080: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000100000 mod=0,1,2,3,4,[ 5, 4 ],[ 5, 6 ],6,7
            size = 9;
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            x[ 8 ] = x_7;
            y[ 8 ] = y_7;
            c[ 8 ] = reinterpret_cast<const CI *>( &pc_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x609080403020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_5 );
            __m128d x_i0 = _mm_set1_pd( x_5 );
            __m128d y_i0 = _mm_set1_pd( y_5 );
            __m128d d_i1 = _mm_set_pd( d_6, d_4 );
            __m128d x_i1 = _mm_set_pd( x_6, x_4 );
            __m128d y_i1 = _mm_set_pd( y_6, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2096: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000110000 mod=0,1,2,3,[ 4, 3 ],[ 5, 6 ],6,7
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x706090803020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_4 );
            __m128d x_i0 = _mm_set_pd( x_5, x_4 );
            __m128d y_i0 = _mm_set_pd( y_5, y_4 );
            __m128d d_i1 = _mm_set_pd( d_6, d_3 );
            __m128d x_i1 = _mm_set_pd( x_6, x_3 );
            __m128d y_i1 = _mm_set_pd( y_6, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2104: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000111000 mod=0,1,2,[ 3, 2 ],[ 5, 6 ],6,7
            size = 7;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x7060908020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_3 );
            __m128d x_i0 = _mm_set_pd( x_5, x_3 );
            __m128d y_i0 = _mm_set_pd( y_5, y_3 );
            __m128d d_i1 = _mm_set_pd( d_6, d_2 );
            __m128d x_i1 = _mm_set_pd( x_6, x_2 );
            __m128d y_i1 = _mm_set_pd( y_6, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2108: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000111100 mod=0,1,[ 2, 1 ],[ 5, 6 ],6,7
            size = 6;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x70609080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_2 );
            __m128d x_i0 = _mm_set_pd( x_5, x_2 );
            __m128d y_i0 = _mm_set_pd( y_5, y_2 );
            __m128d d_i1 = _mm_set_pd( d_6, d_1 );
            __m128d x_i1 = _mm_set_pd( x_6, x_1 );
            __m128d y_i1 = _mm_set_pd( y_6, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2110: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000111110 mod=0,[ 1, 0 ],[ 5, 6 ],6,7
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x706090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_1 );
            __m128d x_i0 = _mm_set_pd( x_5, x_1 );
            __m128d y_i0 = _mm_set_pd( y_5, y_1 );
            __m128d d_i1 = _mm_set_pd( d_6, d_0 );
            __m128d x_i1 = _mm_set_pd( x_6, x_0 );
            __m128d y_i1 = _mm_set_pd( y_6, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2111: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000000111111 mod=[ 0, 7 ],[ 5, 6 ],6,7
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x7060908ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_0 );
            __m128d x_i0 = _mm_set_pd( x_5, x_0 );
            __m128d y_i0 = _mm_set_pd( y_5, y_0 );
            __m128d d_i1 = _mm_set_pd( d_6, d_7 );
            __m128d x_i1 = _mm_set_pd( x_6, x_7 );
            __m128d y_i1 = _mm_set_pd( y_6, y_7 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2112: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000001000000 mod=0,1,2,3,4,5,[ 6, 5 ],[ 6, 7 ],7
            size = 9;
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            x[ 8 ] = x_7;
            y[ 8 ] = y_7;
            c[ 8 ] = reinterpret_cast<const CI *>( &pc_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x908050403020100ul ) );
            __m128d d_i0 = _mm_set1_pd( d_6 );
            __m128d x_i0 = _mm_set1_pd( x_6 );
            __m128d y_i0 = _mm_set1_pd( y_6 );
            __m128d d_i1 = _mm_set_pd( d_7, d_5 );
            __m128d x_i1 = _mm_set_pd( x_7, x_5 );
            __m128d y_i1 = _mm_set_pd( y_7, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2144: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000001100000 mod=0,1,2,3,4,[ 5, 4 ],[ 6, 7 ],7
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x709080403020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_5 );
            __m128d x_i0 = _mm_set_pd( x_6, x_5 );
            __m128d y_i0 = _mm_set_pd( y_6, y_5 );
            __m128d d_i1 = _mm_set_pd( d_7, d_4 );
            __m128d x_i1 = _mm_set_pd( x_7, x_4 );
            __m128d y_i1 = _mm_set_pd( y_7, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2160: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000001110000 mod=0,1,2,3,[ 4, 3 ],[ 6, 7 ],7
            size = 7;
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x7090803020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_4 );
            __m128d x_i0 = _mm_set_pd( x_6, x_4 );
            __m128d y_i0 = _mm_set_pd( y_6, y_4 );
            __m128d d_i1 = _mm_set_pd( d_7, d_3 );
            __m128d x_i1 = _mm_set_pd( x_7, x_3 );
            __m128d y_i1 = _mm_set_pd( y_7, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2168: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000001111000 mod=0,1,2,[ 3, 2 ],[ 6, 7 ],7
            size = 6;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x70908020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_3 );
            __m128d x_i0 = _mm_set_pd( x_6, x_3 );
            __m128d y_i0 = _mm_set_pd( y_6, y_3 );
            __m128d d_i1 = _mm_set_pd( d_7, d_2 );
            __m128d x_i1 = _mm_set_pd( x_7, x_2 );
            __m128d y_i1 = _mm_set_pd( y_7, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2172: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000001111100 mod=0,1,[ 2, 1 ],[ 6, 7 ],7
            size = 5;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x709080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_2 );
            __m128d x_i0 = _mm_set_pd( x_6, x_2 );
            __m128d y_i0 = _mm_set_pd( y_6, y_2 );
            __m128d d_i1 = _mm_set_pd( d_7, d_1 );
            __m128d x_i1 = _mm_set_pd( x_7, x_1 );
            __m128d y_i1 = _mm_set_pd( y_7, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2174: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000001111110 mod=0,[ 1, 0 ],[ 6, 7 ],7
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x7090800ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_1 );
            __m128d x_i0 = _mm_set_pd( x_6, x_1 );
            __m128d y_i0 = _mm_set_pd( y_6, y_1 );
            __m128d d_i1 = _mm_set_pd( d_7, d_0 );
            __m128d x_i1 = _mm_set_pd( x_7, x_0 );
            __m128d y_i1 = _mm_set_pd( y_7, y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2175: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000001111111 mod=[ 0, 7 ],[ 6, 7 ],7
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x70908ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_0 );
            __m128d x_i0 = _mm_set_pd( x_6, x_0 );
            __m128d y_i0 = _mm_set_pd( y_6, y_0 );
            __m128d d_i1 = _mm_set1_pd( d_7 );
            __m128d x_i1 = _mm_set1_pd( x_7 );
            __m128d y_i1 = _mm_set1_pd( y_7 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 6 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2176: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000010000000 mod=0,1,2,3,4,5,6,[ 7, 6 ],[ 7, 0 ]
            size = 9;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            TF m_7_0 = d_7 / ( d_0 - d_7 );
            x[ 8 ] = x_7 - m_7_0 * ( x_0 - x_7 );
            y[ 8 ] = y_7 - m_7_0 * ( y_0 - y_7 );
            c[ 8 ] = reinterpret_cast<const CI *>( &pc_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x806050403020100ul ) );
            TF m = d_7 / ( d_6 - d_7 ); // 1
            __m512d inter_x = _mm512_set1_pd( x_7 - m * ( x_6 - x_7 ) );
            __m512d inter_y = _mm512_set1_pd( y_7 - m * ( y_6 - y_7 ) );
            __m512i inter_c = _mm512_set1_epi64( (long long)cut_id[ num_cut ] );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2177: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000010000001 mod=[ 0, 1 ],1,2,3,4,5,6,[ 7, 6 ]
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x906050403020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_7, d_0 );
            __m128d x_i0 = _mm_set_pd( x_7, x_0 );
            __m128d y_i0 = _mm_set_pd( y_7, y_0 );
            __m128d d_i1 = _mm_set_pd( d_6, d_1 );
            __m128d x_i1 = _mm_set_pd( x_6, x_1 );
            __m128d y_i1 = _mm_set_pd( y_6, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2179: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000010000011 mod=[ 7, 6 ],[ 1, 2 ],2,3,4,5,6
            size = 7;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x6050403020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_7 );
            __m128d x_i0 = _mm_set_pd( x_1, x_7 );
            __m128d y_i0 = _mm_set_pd( y_1, y_7 );
            __m128d d_i1 = _mm_set_pd( d_2, d_6 );
            __m128d x_i1 = _mm_set_pd( x_2, x_6 );
            __m128d y_i1 = _mm_set_pd( y_2, y_6 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2183: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000010000111 mod=6,[ 7, 6 ],[ 2, 3 ],3,4,5
            size = 6;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50403090806ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_7 );
            __m128d x_i0 = _mm_set_pd( x_2, x_7 );
            __m128d y_i0 = _mm_set_pd( y_2, y_7 );
            __m128d d_i1 = _mm_set_pd( d_3, d_6 );
            __m128d x_i1 = _mm_set_pd( x_3, x_6 );
            __m128d y_i1 = _mm_set_pd( y_3, y_6 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2191: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000010001111 mod=5,6,[ 7, 6 ],[ 3, 4 ],4
            size = 5;
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x409080605ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_7 );
            __m128d x_i0 = _mm_set_pd( x_3, x_7 );
            __m128d y_i0 = _mm_set_pd( y_3, y_7 );
            __m128d d_i1 = _mm_set_pd( d_4, d_6 );
            __m128d x_i1 = _mm_set_pd( x_4, x_6 );
            __m128d y_i1 = _mm_set_pd( y_4, y_6 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2207: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000010011111 mod=[ 4, 5 ],5,6,[ 7, 6 ]
            size = 4;
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9060508ul ) );
            __m128d d_i0 = _mm_set_pd( d_7, d_4 );
            __m128d x_i0 = _mm_set_pd( x_7, x_4 );
            __m128d y_i0 = _mm_set_pd( y_7, y_4 );
            __m128d d_i1 = _mm_set_pd( d_6, d_5 );
            __m128d x_i1 = _mm_set_pd( x_6, x_5 );
            __m128d y_i1 = _mm_set_pd( y_6, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2239: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000010111111 mod=[ 5, 6 ],6,[ 7, 6 ]
            size = 3;
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90608ul ) );
            __m128d d_i0 = _mm_set_pd( d_7, d_5 );
            __m128d x_i0 = _mm_set_pd( x_7, x_5 );
            __m128d y_i0 = _mm_set_pd( y_7, y_5 );
            __m128d d_i1 = _mm_set1_pd( d_6 );
            __m128d x_i1 = _mm_set1_pd( x_6 );
            __m128d y_i1 = _mm_set1_pd( y_6 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 5 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2240: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011000000 mod=0,1,2,3,4,5,[ 6, 5 ],[ 7, 0 ]
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x908050403020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_7, d_6 );
            __m128d x_i0 = _mm_set_pd( x_7, x_6 );
            __m128d y_i0 = _mm_set_pd( y_7, y_6 );
            __m128d d_i1 = _mm_set_pd( d_0, d_5 );
            __m128d x_i1 = _mm_set_pd( x_0, x_5 );
            __m128d y_i1 = _mm_set_pd( y_0, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 7 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2241: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011000001 mod=[ 0, 1 ],1,2,3,4,5,[ 6, 5 ]
            size = 7;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9050403020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_0 );
            __m128d x_i0 = _mm_set_pd( x_6, x_0 );
            __m128d y_i0 = _mm_set_pd( y_6, y_0 );
            __m128d d_i1 = _mm_set_pd( d_5, d_1 );
            __m128d x_i1 = _mm_set_pd( x_5, x_1 );
            __m128d y_i1 = _mm_set_pd( y_5, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2243: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011000011 mod=[ 6, 5 ],[ 1, 2 ],2,3,4,5
            size = 6;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x50403020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_6 );
            __m128d x_i0 = _mm_set_pd( x_1, x_6 );
            __m128d y_i0 = _mm_set_pd( y_1, y_6 );
            __m128d d_i1 = _mm_set_pd( d_2, d_5 );
            __m128d x_i1 = _mm_set_pd( x_2, x_5 );
            __m128d y_i1 = _mm_set_pd( y_2, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2247: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011000111 mod=5,[ 6, 5 ],[ 2, 3 ],3,4
            size = 5;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x403090805ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_6 );
            __m128d x_i0 = _mm_set_pd( x_2, x_6 );
            __m128d y_i0 = _mm_set_pd( y_2, y_6 );
            __m128d d_i1 = _mm_set_pd( d_3, d_5 );
            __m128d x_i1 = _mm_set_pd( x_3, x_5 );
            __m128d y_i1 = _mm_set_pd( y_3, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2255: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011001111 mod=4,5,[ 6, 5 ],[ 3, 4 ]
            size = 4;
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9080504ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_6 );
            __m128d x_i0 = _mm_set_pd( x_3, x_6 );
            __m128d y_i0 = _mm_set_pd( y_3, y_6 );
            __m128d d_i1 = _mm_set_pd( d_4, d_5 );
            __m128d x_i1 = _mm_set_pd( x_4, x_5 );
            __m128d y_i1 = _mm_set_pd( y_4, y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2271: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011011111 mod=[ 4, 5 ],5,[ 6, 5 ]
            size = 3;
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_6 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_6 = reinterpret_cast<const TF *>( &px_0 )[ 6 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_6 = reinterpret_cast<const TF *>( &py_0 )[ 6 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90508ul ) );
            __m128d d_i0 = _mm_set_pd( d_6, d_4 );
            __m128d x_i0 = _mm_set_pd( x_6, x_4 );
            __m128d y_i0 = _mm_set_pd( y_6, y_4 );
            __m128d d_i1 = _mm_set1_pd( d_5 );
            __m128d x_i1 = _mm_set1_pd( x_5 );
            __m128d y_i1 = _mm_set1_pd( y_5 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 4 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2272: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011100000 mod=0,1,2,3,4,[ 5, 4 ],[ 7, 0 ]
            size = 7;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9080403020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_7, d_5 );
            __m128d x_i0 = _mm_set_pd( x_7, x_5 );
            __m128d y_i0 = _mm_set_pd( y_7, y_5 );
            __m128d d_i1 = _mm_set_pd( d_0, d_4 );
            __m128d x_i1 = _mm_set_pd( x_0, x_4 );
            __m128d y_i1 = _mm_set_pd( y_0, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 7 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2273: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011100001 mod=[ 0, 1 ],1,2,3,4,[ 5, 4 ]
            size = 6;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90403020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_0 );
            __m128d x_i0 = _mm_set_pd( x_5, x_0 );
            __m128d y_i0 = _mm_set_pd( y_5, y_0 );
            __m128d d_i1 = _mm_set_pd( d_4, d_1 );
            __m128d x_i1 = _mm_set_pd( x_4, x_1 );
            __m128d y_i1 = _mm_set_pd( y_4, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2275: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011100011 mod=[ 5, 4 ],[ 1, 2 ],2,3,4
            size = 5;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x403020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_5 );
            __m128d x_i0 = _mm_set_pd( x_1, x_5 );
            __m128d y_i0 = _mm_set_pd( y_1, y_5 );
            __m128d d_i1 = _mm_set_pd( d_2, d_4 );
            __m128d x_i1 = _mm_set_pd( x_2, x_4 );
            __m128d y_i1 = _mm_set_pd( y_2, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2279: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011100111 mod=4,[ 5, 4 ],[ 2, 3 ],3
            size = 4;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x3090804ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_5 );
            __m128d x_i0 = _mm_set_pd( x_2, x_5 );
            __m128d y_i0 = _mm_set_pd( y_2, y_5 );
            __m128d d_i1 = _mm_set_pd( d_3, d_4 );
            __m128d x_i1 = _mm_set_pd( x_3, x_4 );
            __m128d y_i1 = _mm_set_pd( y_3, y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2287: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011101111 mod=[ 3, 4 ],4,[ 5, 4 ]
            size = 3;
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_5 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_5 = reinterpret_cast<const TF *>( &px_0 )[ 5 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_5 = reinterpret_cast<const TF *>( &py_0 )[ 5 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90408ul ) );
            __m128d d_i0 = _mm_set_pd( d_5, d_3 );
            __m128d x_i0 = _mm_set_pd( x_5, x_3 );
            __m128d y_i0 = _mm_set_pd( y_5, y_3 );
            __m128d d_i1 = _mm_set1_pd( d_4 );
            __m128d x_i1 = _mm_set1_pd( x_4 );
            __m128d y_i1 = _mm_set1_pd( y_4 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 3 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2288: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011110000 mod=0,1,2,3,[ 4, 3 ],[ 7, 0 ]
            size = 6;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90803020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_7, d_4 );
            __m128d x_i0 = _mm_set_pd( x_7, x_4 );
            __m128d y_i0 = _mm_set_pd( y_7, y_4 );
            __m128d d_i1 = _mm_set_pd( d_0, d_3 );
            __m128d x_i1 = _mm_set_pd( x_0, x_3 );
            __m128d y_i1 = _mm_set_pd( y_0, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 7 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2289: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011110001 mod=[ 0, 1 ],1,2,3,[ 4, 3 ]
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x903020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_4, d_0 );
            __m128d x_i0 = _mm_set_pd( x_4, x_0 );
            __m128d y_i0 = _mm_set_pd( y_4, y_0 );
            __m128d d_i1 = _mm_set_pd( d_3, d_1 );
            __m128d x_i1 = _mm_set_pd( x_3, x_1 );
            __m128d y_i1 = _mm_set_pd( y_3, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2291: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011110011 mod=[ 4, 3 ],[ 1, 2 ],2,3
            size = 4;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x3020908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_4 );
            __m128d x_i0 = _mm_set_pd( x_1, x_4 );
            __m128d y_i0 = _mm_set_pd( y_1, y_4 );
            __m128d d_i1 = _mm_set_pd( d_2, d_3 );
            __m128d x_i1 = _mm_set_pd( x_2, x_3 );
            __m128d y_i1 = _mm_set_pd( y_2, y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2295: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011110111 mod=3,[ 4, 3 ],[ 2, 3 ]
            size = 3;
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_4 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_4 = reinterpret_cast<const TF *>( &px_0 )[ 4 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_4 = reinterpret_cast<const TF *>( &py_0 )[ 4 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90803ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_4 );
            __m128d x_i0 = _mm_set_pd( x_2, x_4 );
            __m128d y_i0 = _mm_set_pd( y_2, y_4 );
            __m128d d_i1 = _mm_set1_pd( d_3 );
            __m128d x_i1 = _mm_set1_pd( x_3 );
            __m128d y_i1 = _mm_set1_pd( y_3 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 2 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2296: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011111000 mod=0,1,2,[ 3, 2 ],[ 7, 0 ]
            size = 5;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x908020100ul ) );
            __m128d d_i0 = _mm_set_pd( d_7, d_3 );
            __m128d x_i0 = _mm_set_pd( x_7, x_3 );
            __m128d y_i0 = _mm_set_pd( y_7, y_3 );
            __m128d d_i1 = _mm_set_pd( d_0, d_2 );
            __m128d x_i1 = _mm_set_pd( x_0, x_2 );
            __m128d y_i1 = _mm_set_pd( y_0, y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 7 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2297: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011111001 mod=[ 0, 1 ],1,2,[ 3, 2 ]
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9020108ul ) );
            __m128d d_i0 = _mm_set_pd( d_3, d_0 );
            __m128d x_i0 = _mm_set_pd( x_3, x_0 );
            __m128d y_i0 = _mm_set_pd( y_3, y_0 );
            __m128d d_i1 = _mm_set_pd( d_2, d_1 );
            __m128d x_i1 = _mm_set_pd( x_2, x_1 );
            __m128d y_i1 = _mm_set_pd( y_2, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2299: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011111011 mod=[ 3, 2 ],[ 1, 2 ],2
            size = 3;
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_3 = reinterpret_cast<const TF *>( &px_0 )[ 3 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_3 = reinterpret_cast<const TF *>( &py_0 )[ 3 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x20908ul ) );
            __m128d d_i0 = _mm_set_pd( d_1, d_3 );
            __m128d x_i0 = _mm_set_pd( x_1, x_3 );
            __m128d y_i0 = _mm_set_pd( y_1, y_3 );
            __m128d d_i1 = _mm_set1_pd( d_2 );
            __m128d x_i1 = _mm_set1_pd( x_2 );
            __m128d y_i1 = _mm_set1_pd( y_2 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 1 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2300: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011111100 mod=0,1,[ 2, 1 ],[ 7, 0 ]
            size = 4;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x9080100ul ) );
            __m128d d_i0 = _mm_set_pd( d_7, d_2 );
            __m128d x_i0 = _mm_set_pd( x_7, x_2 );
            __m128d y_i0 = _mm_set_pd( y_7, y_2 );
            __m128d d_i1 = _mm_set_pd( d_0, d_1 );
            __m128d x_i1 = _mm_set_pd( x_0, x_1 );
            __m128d y_i1 = _mm_set_pd( y_0, y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 7 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2301: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011111101 mod=[ 0, 1 ],1,[ 2, 1 ]
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_2 = reinterpret_cast<const TF *>( &px_0 )[ 2 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_2 = reinterpret_cast<const TF *>( &py_0 )[ 2 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90108ul ) );
            __m128d d_i0 = _mm_set_pd( d_2, d_0 );
            __m128d x_i0 = _mm_set_pd( x_2, x_0 );
            __m128d y_i0 = _mm_set_pd( y_2, y_0 );
            __m128d d_i1 = _mm_set1_pd( d_1 );
            __m128d x_i1 = _mm_set1_pd( x_1 );
            __m128d y_i1 = _mm_set1_pd( y_1 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)cut_id[ num_cut ], (long long)reinterpret_cast<const CI *>( &pc_0 )[ 0 ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 2302: {
            // size=8 outside=0000000000000000000000000000000000000000000000000000000011111110 mod=0,[ 1, 0 ],[ 7, 0 ]
            size = 3;
            TF d_0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
            TF d_1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
            TF d_7 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
            TF x_0 = reinterpret_cast<const TF *>( &px_0 )[ 0 ];
            TF x_1 = reinterpret_cast<const TF *>( &px_0 )[ 1 ];
            TF x_7 = reinterpret_cast<const TF *>( &px_0 )[ 7 ];
            TF y_0 = reinterpret_cast<const TF *>( &py_0 )[ 0 ];
            TF y_1 = reinterpret_cast<const TF *>( &py_0 )[ 1 ];
            TF y_7 = reinterpret_cast<const TF *>( &py_0 )[ 7 ];
            __m512i idx_0 = _mm512_cvtepu8_epi64( _mm_cvtsi64_si128( 0x90800ul ) );
            __m128d d_i0 = _mm_set_pd( d_7, d_1 );
            __m128d x_i0 = _mm_set_pd( x_7, x_1 );
            __m128d y_i0 = _mm_set_pd( y_7, y_1 );
            __m128d d_i1 = _mm_set1_pd( d_0 );
            __m128d x_i1 = _mm_set1_pd( x_0 );
            __m128d y_i1 = _mm_set1_pd( y_0 );
            __m128d m = _mm_div_pd( d_i0, _mm_sub_pd( d_i1, d_i0 ) );
            __m512d inter_x = _mm512_castpd128_pd512( _mm_sub_pd( x_i0, _mm_mul_pd( m, _mm_sub_pd( x_i1, x_i0 ) ) ) );
            __m512d inter_y = _mm512_castpd128_pd512( _mm_sub_pd( y_i0, _mm_mul_pd( m, _mm_sub_pd( y_i1, y_i0 ) ) ) );
            __m512i inter_c = _mm512_castsi128_si512( _mm_set_epi64x( (long long)reinterpret_cast<const CI *>( &pc_0 )[ 7 ], (long long)cut_id[ num_cut ] ) );
            px_0 = _mm512_permutex2var_pd( px_0, idx_0, inter_x );
            py_0 = _mm512_permutex2var_pd( py_0, idx_0, inter_y );
            pc_0 = _mm512_permutex2var_epi64( pc_0, idx_0, inter_c );
            break;
        }
        case 0:
        case 1:
        case 2:
        case 3:
        case 4:
        case 5:
        case 6:
        case 7:
        case 8:
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
        case 16:
        case 17:
        case 18:
        case 19:
        case 20:
        case 21:
        case 22:
        case 23:
        case 24:
        case 25:
        case 26:
        case 27:
        case 28:
        case 29:
        case 30:
        case 31:
        case 32:
        case 33:
        case 34:
        case 35:
        case 36:
        case 37:
        case 38:
        case 39:
        case 40:
        case 41:
        case 42:
        case 43:
        case 44:
        case 45:
        case 46:
        case 47:
        case 48:
        case 49:
        case 50:
        case 51:
        case 52:
        case 53:
        case 54:
        case 55:
        case 56:
        case 57:
        case 58:
        case 59:
        case 60:
        case 61:
        case 62:
        case 63:
        case 64:
        case 65:
        case 66:
        case 67:
        case 68:
        case 69:
        case 70:
        case 71:
        case 72:
        case 73:
        case 74:
        case 75:
        case 76:
        case 77:
        case 78:
        case 79:
        case 80:
        case 81:
        case 82:
        case 83:
        case 84:
        case 85:
        case 86:
        case 87:
        case 88:
        case 89:
        case 90:
        case 91:
        case 92:
        case 93:
        case 94:
        case 95:
        case 96:
        case 97:
        case 98:
        case 99:
        case 100:
        case 101:
        case 102:
        case 103:
        case 104:
        case 105:
        case 106:
        case 107:
        case 108:
        case 109:
        case 110:
        case 111:
        case 112:
        case 113:
        case 114:
        case 115:
        case 116:
        case 117:
        case 118:
        case 119:
        case 120:
        case 121:
        case 122:
        case 123:
        case 124:
        case 125:
        case 126:
        case 127:
        case 128:
        case 129:
        case 130:
        case 131:
        case 132:
        case 133:
        case 134:
        case 135:
        case 136:
        case 137:
        case 138:
        case 139:
        case 140:
        case 141:
        case 142:
        case 143:
        case 144:
        case 145:
        case 146:
        case 147:
        case 148:
        case 149:
        case 150:
        case 151:
        case 152:
        case 153:
        case 154:
        case 155:
        case 156:
        case 157:
        case 158:
        case 159:
        case 160:
        case 161:
        case 162:
        case 163:
        case 164:
        case 165:
        case 166:
        case 167:
        case 168:
        case 169:
        case 170:
        case 171:
        case 172:
        case 173:
        case 174:
        case 175:
        case 176:
        case 177:
        case 178:
        case 179:
        case 180:
        case 181:
        case 182:
        case 183:
        case 184:
        case 185:
        case 186:
        case 187:
        case 188:
        case 189:
        case 190:
        case 191:
        case 192:
        case 193:
        case 194:
        case 195:
        case 196:
        case 197:
        case 198:
        case 199:
        case 200:
        case 201:
        case 202:
        case 203:
        case 204:
        case 205:
        case 206:
        case 207:
        case 208:
        case 209:
        case 210:
        case 211:
        case 212:
        case 213:
        case 214:
        case 215:
        case 216:
        case 217:
        case 218:
        case 219:
        case 220:
        case 221:
        case 222:
        case 223:
        case 224:
        case 225:
        case 226:
        case 227:
        case 228:
        case 229:
        case 230:
        case 231:
        case 232:
        case 233:
        case 234:
        case 235:
        case 236:
        case 237:
        case 238:
        case 239:
        case 240:
        case 241:
        case 242:
        case 243:
        case 244:
        case 245:
        case 246:
        case 247:
        case 248:
        case 249:
        case 250:
        case 251:
        case 252:
        case 253:
        case 254:
        case 255:
        case 256:
        case 258:
        case 260:
        case 262:
        case 264:
        case 266:
        case 268:
        case 270:
        case 272:
        case 274:
        case 276:
        case 278:
        case 280:
        case 282:
        case 284:
        case 286:
        case 288:
        case 290:
        case 292:
        case 294:
        case 296:
        case 298:
        case 300:
        case 302:
        case 304:
        case 306:
        case 308:
        case 310:
        case 312:
        case 314:
        case 316:
        case 318:
        case 320:
        case 322:
        case 324:
        case 326:
        case 328:
        case 330:
        case 332:
        case 334:
        case 336:
        case 338:
        case 340:
        case 342:
        case 344:
        case 346:
        case 348:
        case 350:
        case 352:
        case 354:
        case 356:
        case 358:
        case 360:
        case 362:
        case 364:
        case 366:
        case 368:
        case 370:
        case 372:
        case 374:
        case 376:
        case 378:
        case 380:
        case 382:
        case 384:
        case 386:
        case 388:
        case 390:
        case 392:
        case 394:
        case 396:
        case 398:
        case 400:
        case 402:
        case 404:
        case 406:
        case 408:
        case 410:
        case 412:
        case 414:
        case 416:
        case 418:
        case 420:
        case 422:
        case 424:
        case 426:
        case 428:
        case 430:
        case 432:
        case 434:
        case 436:
        case 438:
        case 440:
        case 442:
        case 444:
        case 446:
        case 448:
        case 450:
        case 452:
        case 454:
        case 456:
        case 458:
        case 460:
        case 462:
        case 464:
        case 466:
        case 468:
        case 470:
        case 472:
        case 474:
        case 476:
        case 478:
        case 480:
        case 482:
        case 484:
        case 486:
        case 488:
        case 490:
        case 492:
        case 494:
        case 496:
        case 498:
        case 500:
        case 502:
        case 504:
        case 506:
        case 508:
        case 510:
        case 512:
        case 516:
        case 520:
        case 524:
        case 528:
        case 532:
        case 536:
        case 540:
        case 544:
        case 548:
        case 552:
        case 556:
        case 560:
        case 564:
        case 568:
        case 572:
        case 576:
        case 580:
        case 584:
        case 588:
        case 592:
        case 596:
        case 600:
        case 604:
        case 608:
        case 612:
        case 616:
        case 620:
        case 624:
        case 628:
        case 632:
        case 636:
        case 640:
        case 644:
        case 648:
        case 652:
        case 656:
        case 660:
        case 664:
        case 668:
        case 672:
        case 676:
        case 680:
        case 684:
        case 688:
        case 692:
        case 696:
        case 700:
        case 704:
        case 708:
        case 712:
        case 716:
        case 720:
        case 724:
        case 728:
        case 732:
        case 736:
        case 740:
        case 744:
        case 748:
        case 752:
        case 756:
        case 760:
        case 764:
        case 768:
        case 776:
        case 784:
        case 792:
        case 800:
        case 808:
        case 816:
        case 824:
        case 832:
        case 840:
        case 848:
        case 856:
        case 864:
        case 872:
        case 880:
        case 888:
        case 896:
        case 904:
        case 912:
        case 920:
        case 928:
        case 936:
        case 944:
        case 952:
        case 960:
        case 968:
        case 976:
        case 984:
        case 992:
        case 1000:
        case 1008:
        case 1016:
        case 1024:
        case 1040:
        case 1056:
        case 1072:
        case 1088:
        case 1104:
        case 1120:
        case 1136:
        case 1152:
        case 1168:
        case 1184:
        case 1200:
        case 1216:
        case 1232:
        case 1248:
        case 1264:
        case 1280:
        case 1312:
        case 1344:
        case 1376:
        case 1408:
        case 1440:
        case 1472:
        case 1504:
        case 1536:
        case 1600:
        case 1664:
        case 1728:
        case 1792:
        case 1920:
        case 2048: {
            break;
        }
        case 257:
        case 259:
        case 261:
        case 263:
        case 265:
        case 267:
        case 269:
        case 271:
        case 273:
        case 275:
        case 277:
        case 279:
        case 281:
        case 283:
        case 285:
        case 287:
        case 289:
        case 291:
        case 293:
        case 295:
        case 297:
        case 299:
        case 301:
        case 303:
        case 305:
        case 307:
        case 309:
        case 311:
        case 313:
        case 315:
        case 317:
        case 319:
        case 321:
        case 323:
        case 325:
        case 327:
        case 329:
        case 331:
        case 333:
        case 335:
        case 337:
        case 339:
        case 341:
        case 343:
        case 345:
        case 347:
        case 349:
        case 351:
        case 353:
        case 355:
        case 357:
        case 359:
        case 361:
        case 363:
        case 365:
        case 367:
        case 369:
        case 371:
        case 373:
        case 375:
        case 377:
        case 379:
        case 381:
        case 383:
        case 385:
        case 387:
        case 389:
        case 391:
        case 393:
        case 395:
        case 397:
        case 399:
        case 401:
        case 403:
        case 405:
        case 407:
        case 409:
        case 411:
        case 413:
        case 415:
        case 417:
        case 419:
        case 421:
        case 423:
        case 425:
        case 427:
        case 429:
        case 431:
        case 433:
        case 435:
        case 437:
        case 439:
        case 441:
        case 443:
        case 445:
        case 447:
        case 449:
        case 451:
        case 453:
        case 455:
        case 457:
        case 459:
        case 461:
        case 463:
        case 465:
        case 467:
        case 469:
        case 471:
        case 473:
        case 475:
        case 477:
        case 479:
        case 481:
        case 483:
        case 485:
        case 487:
        case 489:
        case 491:
        case 493:
        case 495:
        case 497:
        case 499:
        case 501:
        case 503:
        case 505:
        case 507:
        case 509:
        case 511:
        case 513:
        case 514:
        case 515:
        case 517:
        case 518:
        case 519:
        case 521:
        case 522:
        case 523:
        case 525:
        case 526:
        case 527:
        case 529:
        case 530:
        case 531:
        case 533:
        case 534:
        case 535:
        case 537:
        case 538:
        case 539:
        case 541:
        case 542:
        case 543:
        case 545:
        case 546:
        case 547:
        case 549:
        case 550:
        case 551:
        case 553:
        case 554:
        case 555:
        case 557:
        case 558:
        case 559:
        case 561:
        case 562:
        case 563:
        case 565:
        case 566:
        case 567:
        case 569:
        case 570:
        case 571:
        case 573:
        case 574:
        case 575:
        case 577:
        case 578:
        case 579:
        case 581:
        case 582:
        case 583:
        case 585:
        case 586:
        case 587:
        case 589:
        case 590:
        case 591:
        case 593:
        case 594:
        case 595:
        case 597:
        case 598:
        case 599:
        case 601:
        case 602:
        case 603:
        case 605:
        case 606:
        case 607:
        case 609:
        case 610:
        case 611:
        case 613:
        case 614:
        case 615:
        case 617:
        case 618:
        case 619:
        case 621:
        case 622:
        case 623:
        case 625:
        case 626:
        case 627:
        case 629:
        case 630:
        case 631:
        case 633:
        case 634:
        case 635:
        case 637:
        case 638:
        case 639:
        case 641:
        case 642:
        case 643:
        case 645:
        case 646:
        case 647:
        case 649:
        case 650:
        case 651:
        case 653:
        case 654:
        case 655:
        case 657:
        case 658:
        case 659:
        case 661:
        case 662:
        case 663:
        case 665:
        case 666:
        case 667:
        case 669:
        case 670:
        case 671:
        case 673:
        case 674:
        case 675:
        case 677:
        case 678:
        case 679:
        case 681:
        case 682:
        case 683:
        case 685:
        case 686:
        case 687:
        case 689:
        case 690:
        case 691:
        case 693:
        case 694:
        case 695:
        case 697:
        case 698:
        case 699:
        case 701:
        case 702:
        case 703:
        case 705:
        case 706:
        case 707:
        case 709:
        case 710:
        case 711:
        case 713:
        case 714:
        case 715:
        case 717:
        case 718:
        case 719:
        case 721:
        case 722:
        case 723:
        case 725:
        case 726:
        case 727:
        case 729:
        case 730:
        case 731:
        case 733:
        case 734:
        case 735:
        case 737:
        case 738:
        case 739:
        case 741:
        case 742:
        case 743:
        case 745:
        case 746:
        case 747:
        case 749:
        case 750:
        case 751:
        case 753:
        case 754:
        case 755:
        case 757:
        case 758:
        case 759:
        case 761:
        case 762:
        case 763:
        case 765:
        case 766:
        case 767:
        case 775:
        case 783:
        case 791:
        case 799:
        case 807:
        case 815:
        case 823:
        case 831:
        case 839:
        case 847:
        case 855:
        case 863:
        case 871:
        case 879:
        case 887:
        case 895:
        case 903:
        case 911:
        case 919:
        case 927:
        case 935:
        case 943:
        case 951:
        case 959:
        case 967:
        case 975:
        case 983:
        case 991:
        case 999:
        case 1007:
        case 1015:
        case 1023:
        case 1039:
        case 1055:
        case 1071:
        case 1087:
        case 1103:
        case 1119:
        case 1135:
        case 1151:
        case 1167:
        case 1183:
        case 1199:
        case 1215:
        case 1231:
        case 1247:
        case 1263:
        case 1279:
        case 1311:
        case 1343:
        case 1375:
        case 1407:
        case 1439:
        case 1471:
        case 1503:
        case 1535:
        case 1599:
        case 1663:
        case 1727:
        case 1791:
        case 1919:
        case 2047:
        case 2303: {
            size = 0;
            break;
        }
        default:
            _mm512_store_pd( x + 0, px_0 );
            _mm512_store_pd( y + 0, py_0 );
            _mm512_store_epi64( c + 0, pc_0 );
            plane_cut_gen( cut_dir[ 0 ][ num_cut ], cut_dir[ 1 ][ num_cut ], cut_ps[ num_cut ], cut_id[ num_cut ], N<flags>() );
            x = &nodes->x;
            y = &nodes->y;
            c = &nodes->cut_id.val;
            px_0 = _mm512_load_pd( x + 0 );
            py_0 = _mm512_load_pd( y + 0 );
            pc_0 = _mm512_load_epi64( c + 0 );
            break;
        }
    }
    _mm512_store_pd( x + 0, px_0 );
    _mm512_store_pd( y + 0, py_0 );
    _mm512_store_epi64( c + 0, pc_0 );
    #else // __AVX512F__
    for( std::size_t i = 0; i < nb_cuts; ++i )
        plane_cut_gen( cut_dir[ 0 ][ i ], cut_dir[ 1 ][ i ], cut_ps[ i ], cut_id[ i ], N<flags>() );
    #endif // __AVX512F__
}

} // namespace sdot
