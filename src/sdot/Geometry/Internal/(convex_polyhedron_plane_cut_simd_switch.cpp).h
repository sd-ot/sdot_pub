#include "../ConvexPolyhedron2.h"
namespace sdot {

template<class Pc> template<int flags>
bool ConvexPolyhedron2<Pc>::plane_cut_simd_switch( Pt origin, Pt normal, CI cut_id, N<flags>, S<double> ) {
    #ifdef __AVX512F__
    // outsize list
    __m512d ox = _mm512_set1_pd( origin.x );
    __m512d oy = _mm512_set1_pd( origin.y );
    __m512d nx = _mm512_set1_pd( normal.x );
    __m512d ny = _mm512_set1_pd( normal.y );
    __m512d px_0 = _mm512_load_pd( &nodes->x + 0 );
    __m512d py_0 = _mm512_load_pd( &nodes->y + 0 );
    __m512d di_0 = _mm512_add_pd( _mm512_mul_pd( _mm512_sub_pd( ox, px_0 ), nx ), _mm512_mul_pd( _mm512_sub_pd( oy, py_0 ), ny ) );
    std::uint8_t outside_0 = _mm512_cmp_pd_mask( di_0, _mm512_set1_pd( 0.0 ), _CMP_LT_OQ ); // OS?
    
    switch( 256 * size + 1 * outside_0 ) {
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 2 );
        Node &n3 = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1030:
    case 1034:
    case 1046:
    case 1050:
    case 1062:
    case 1066:
    case 1078:
    case 1082:
    case 1094:
    case 1098:
    case 1110:
    case 1114:
    case 1126:
    case 1130:
    case 1142:
    case 1146:
    case 1158:
    case 1162:
    case 1174:
    case 1178:
    case 1190:
    case 1194:
    case 1206:
    case 1210:
    case 1222:
    case 1226:
    case 1238:
    case 1242:
    case 1254:
    case 1258:
    case 1270:
    case 1274:
    case 1286:
    case 1290:
    case 1298:
    case 1318:
    case 1322:
    case 1330:
    case 1350:
    case 1354:
    case 1362:
    case 1382:
    case 1386:
    case 1394:
    case 1414:
    case 1418:
    case 1426:
    case 1446:
    case 1450:
    case 1458:
    case 1478:
    case 1482:
    case 1490:
    case 1510:
    case 1514:
    case 1522:
    case 1542:
    case 1546:
    case 1554:
    case 1570:
    case 1606:
    case 1610:
    case 1618:
    case 1634:
    case 1670:
    case 1674:
    case 1682:
    case 1698:
    case 1734:
    case 1738:
    case 1746:
    case 1762:
    case 1798:
    case 1802:
    case 1810:
    case 1826:
    case 1858:
    case 1926:
    case 1930:
    case 1938:
    case 1954:
    case 1986:
    case 2054:
    case 2058:
    case 2066:
    case 2082:
    case 2114:
    case 2178: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 2 );
        Node &n3 = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 3 );
        Node &n3 = node( 0 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 3;
        return true;
    }
    case 2062:
    case 2070:
    case 2074:
    case 2086:
    case 2090:
    case 2098:
    case 2118:
    case 2122:
    case 2130:
    case 2146:
    case 2182:
    case 2186:
    case 2194:
    case 2210:
    case 2242: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 3 );
        Node &n3 = node( 4 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 6 ) );
        nodes->local_at( 6 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 7;
        return true;
    }
    case 1806:
    case 1814:
    case 1818:
    case 1830:
    case 1834:
    case 1842:
    case 1862:
    case 1866:
    case 1874:
    case 1890:
    case 1934:
    case 1942:
    case 1946:
    case 1958:
    case 1962:
    case 1970:
    case 1990:
    case 1994:
    case 2002:
    case 2018: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 3 );
        Node &n3 = node( 4 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 6 ) );
        size = 6;
        return true;
    }
    case 1550:
    case 1558:
    case 1562:
    case 1574:
    case 1578:
    case 1586:
    case 1614:
    case 1622:
    case 1626:
    case 1638:
    case 1642:
    case 1650:
    case 1678:
    case 1686:
    case 1690:
    case 1702:
    case 1706:
    case 1714:
    case 1742:
    case 1750:
    case 1754:
    case 1766:
    case 1770:
    case 1778: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 3 );
        Node &n3 = node( 4 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 5 ) );
        size = 5;
        return true;
    }
    case 1294:
    case 1302:
    case 1306:
    case 1326:
    case 1334:
    case 1338:
    case 1358:
    case 1366:
    case 1370:
    case 1390:
    case 1398:
    case 1402:
    case 1422:
    case 1430:
    case 1434:
    case 1454:
    case 1462:
    case 1466:
    case 1486:
    case 1494:
    case 1498:
    case 1518:
    case 1526:
    case 1530: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 3 );
        Node &n3 = node( 4 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 4 ) );
        size = 4;
        return true;
    }
    case 1310:
    case 1342:
    case 1374:
    case 1406:
    case 1438:
    case 1470:
    case 1502:
    case 1534: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 4 );
        Node &n3 = node( 0 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 3;
        return true;
    }
    case 2078:
    case 2094:
    case 2102:
    case 2106:
    case 2126:
    case 2134:
    case 2138:
    case 2150:
    case 2154:
    case 2162:
    case 2190:
    case 2198:
    case 2202:
    case 2214:
    case 2218:
    case 2226:
    case 2246:
    case 2250:
    case 2258:
    case 2274: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 4 );
        Node &n3 = node( 5 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 6 ) );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 6;
        return true;
    }
    case 1822:
    case 1838:
    case 1846:
    case 1850:
    case 1870:
    case 1878:
    case 1882:
    case 1894:
    case 1898:
    case 1906:
    case 1950:
    case 1966:
    case 1974:
    case 1978:
    case 1998:
    case 2006:
    case 2010:
    case 2022:
    case 2026:
    case 2034: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 4 );
        Node &n3 = node( 5 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 6 ) );
        size = 5;
        return true;
    }
    case 1566:
    case 1582:
    case 1590:
    case 1594:
    case 1630:
    case 1646:
    case 1654:
    case 1658:
    case 1694:
    case 1710:
    case 1718:
    case 1722:
    case 1758:
    case 1774:
    case 1782:
    case 1786: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 4 );
        Node &n3 = node( 5 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 5 ) );
        size = 4;
        return true;
    }
    case 1598:
    case 1662:
    case 1726:
    case 1790: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 5 );
        Node &n3 = node( 0 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 3;
        return true;
    }
    case 2110:
    case 2142:
    case 2158:
    case 2166:
    case 2170:
    case 2206:
    case 2222:
    case 2230:
    case 2234:
    case 2254:
    case 2262:
    case 2266:
    case 2278:
    case 2282:
    case 2290: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 5 );
        Node &n3 = node( 6 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 6 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 5;
        return true;
    }
    case 1854:
    case 1886:
    case 1902:
    case 1910:
    case 1914:
    case 1982:
    case 2014:
    case 2030:
    case 2038:
    case 2042: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 5 );
        Node &n3 = node( 6 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 6 ) );
        size = 4;
        return true;
    }
    case 1918:
    case 2046: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 6 );
        Node &n3 = node( 0 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 3;
        return true;
    }
    case 2174:
    case 2238:
    case 2270:
    case 2286:
    case 2294:
    case 2298: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 6 );
        Node &n3 = node( 7 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 4;
        return true;
    }
    case 2302: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 7 );
        Node &n3 = node( 0 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 3;
        return true;
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
    case 1277:
    case 1309:
    case 1341:
    case 1373:
    case 1405:
    case 1437:
    case 1469:
    case 1501:
    case 1533:
    case 1597:
    case 1661:
    case 1725:
    case 1789:
    case 1917:
    case 2045:
    case 2301: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 0 );
        Node &n3 = node( 1 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 1 ) );
        Node &no = node( 2 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 3;
        return true;
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 0 );
        Node &n3 = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 3 );
        Node &n3 = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1292:
    case 1300:
    case 1324:
    case 1332:
    case 1356:
    case 1364:
    case 1388:
    case 1396:
    case 1420:
    case 1428:
    case 1452:
    case 1460:
    case 1484:
    case 1492:
    case 1516:
    case 1524:
    case 1548:
    case 1556:
    case 1572:
    case 1612:
    case 1620:
    case 1636:
    case 1676:
    case 1684:
    case 1700:
    case 1740:
    case 1748:
    case 1764:
    case 1804:
    case 1812:
    case 1828:
    case 1860:
    case 1932:
    case 1940:
    case 1956:
    case 1988:
    case 2060:
    case 2068:
    case 2084:
    case 2116:
    case 2180: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 3 );
        Node &n3 = node( 4 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1308:
    case 1340:
    case 1372:
    case 1404:
    case 1436:
    case 1468:
    case 1500:
    case 1532: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 4 );
        Node &n3 = node( 0 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 4;
        return true;
    }
    case 2076:
    case 2092:
    case 2100:
    case 2124:
    case 2132:
    case 2148:
    case 2188:
    case 2196:
    case 2212:
    case 2244: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 4 );
        Node &n3 = node( 5 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 6 ) );
        nodes->local_at( 6 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 7;
        return true;
    }
    case 1820:
    case 1836:
    case 1844:
    case 1868:
    case 1876:
    case 1892:
    case 1948:
    case 1964:
    case 1972:
    case 1996:
    case 2004:
    case 2020: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 4 );
        Node &n3 = node( 5 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 6 ) );
        size = 6;
        return true;
    }
    case 1564:
    case 1580:
    case 1588:
    case 1628:
    case 1644:
    case 1652:
    case 1692:
    case 1708:
    case 1716:
    case 1756:
    case 1772:
    case 1780: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 4 );
        Node &n3 = node( 5 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 5 ) );
        size = 5;
        return true;
    }
    case 1596:
    case 1660:
    case 1724:
    case 1788: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 5 );
        Node &n3 = node( 0 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 4;
        return true;
    }
    case 2108:
    case 2140:
    case 2156:
    case 2164:
    case 2204:
    case 2220:
    case 2228:
    case 2252:
    case 2260:
    case 2276: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 5 );
        Node &n3 = node( 6 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 6 ) );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 6;
        return true;
    }
    case 1852:
    case 1884:
    case 1900:
    case 1908:
    case 1980:
    case 2012:
    case 2028:
    case 2036: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 5 );
        Node &n3 = node( 6 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 6 ) );
        size = 5;
        return true;
    }
    case 1916:
    case 2044: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 6 );
        Node &n3 = node( 0 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 4;
        return true;
    }
    case 2172:
    case 2236:
    case 2268:
    case 2284:
    case 2292: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 6 );
        Node &n3 = node( 7 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 5;
        return true;
    }
    case 2300: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 7 );
        Node &n3 = node( 0 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 4;
        return true;
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 2 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &n3 = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1301:
    case 1305:
    case 1333:
    case 1337:
    case 1365:
    case 1369:
    case 1397:
    case 1401:
    case 1429:
    case 1433:
    case 1461:
    case 1465:
    case 1493:
    case 1497:
    case 1525:
    case 1529:
    case 1581:
    case 1589:
    case 1593:
    case 1645:
    case 1653:
    case 1657:
    case 1709:
    case 1717:
    case 1721:
    case 1773:
    case 1781:
    case 1785:
    case 1885:
    case 1901:
    case 1909:
    case 1913:
    case 2013:
    case 2029:
    case 2037:
    case 2041:
    case 2237:
    case 2269:
    case 2285:
    case 2293:
    case 2297: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 0 );
        Node &n3 = node( 1 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 1 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 2 ) );
        Node &no = node( 3 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 4;
        return true;
    }
    case 1029:
    case 1033:
    case 1045:
    case 1049:
    case 1061:
    case 1065:
    case 1077:
    case 1081:
    case 1093:
    case 1097:
    case 1109:
    case 1113:
    case 1125:
    case 1129:
    case 1141:
    case 1145:
    case 1157:
    case 1161:
    case 1173:
    case 1177:
    case 1189:
    case 1193:
    case 1205:
    case 1209:
    case 1221:
    case 1225:
    case 1237:
    case 1241:
    case 1253:
    case 1257:
    case 1269:
    case 1273: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 0 );
        Node &n3 = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
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
    case 1275:
    case 1307:
    case 1339:
    case 1371:
    case 1403:
    case 1435:
    case 1467:
    case 1499:
    case 1531:
    case 1595:
    case 1659:
    case 1723:
    case 1787:
    case 1915:
    case 2043:
    case 2299: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 1 );
        Node &n3 = node( 2 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 2 ) );
        Node &no = node( 2 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 3;
        return true;
    }
    case 1304:
    case 1336:
    case 1368:
    case 1400:
    case 1432:
    case 1464:
    case 1496:
    case 1528: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 4 );
        Node &n3 = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1560:
    case 1576:
    case 1624:
    case 1640:
    case 1688:
    case 1704:
    case 1752:
    case 1768:
    case 1816:
    case 1832:
    case 1864:
    case 1944:
    case 1960:
    case 1992:
    case 2072:
    case 2088:
    case 2120:
    case 2184: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 4 );
        Node &n3 = node( 5 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1592:
    case 1656:
    case 1720:
    case 1784: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 5 );
        Node &n3 = node( 0 );
        Node &nn = node( 4 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 5;
        return true;
    }
    case 2104:
    case 2136:
    case 2152:
    case 2200:
    case 2216:
    case 2248: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 5 );
        Node &n3 = node( 6 );
        Node &nn = node( 4 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 6 ) );
        nodes->local_at( 6 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 7;
        return true;
    }
    case 1848:
    case 1880:
    case 1896:
    case 1976:
    case 2008:
    case 2024: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 5 );
        Node &n3 = node( 6 );
        Node &nn = node( 4 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 6 ) );
        size = 6;
        return true;
    }
    case 1912:
    case 2040: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 6 );
        Node &n3 = node( 0 );
        Node &nn = node( 4 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 5;
        return true;
    }
    case 2168:
    case 2232:
    case 2264:
    case 2280: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 6 );
        Node &n3 = node( 7 );
        Node &nn = node( 4 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 6;
        return true;
    }
    case 2296: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 7 );
        Node &n3 = node( 0 );
        Node &nn = node( 4 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 5;
        return true;
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 3 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &n3 = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 3 );
        Node &n1 = node( 0 );
        Node &n2 = node( 2 );
        Node &n3 = node( 3 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 3 ) );
        size = 3;
        return true;
    }
    case 1573:
    case 1577:
    case 1585:
    case 1637:
    case 1641:
    case 1649:
    case 1701:
    case 1705:
    case 1713:
    case 1765:
    case 1769:
    case 1777:
    case 1869:
    case 1877:
    case 1881:
    case 1893:
    case 1897:
    case 1905:
    case 1997:
    case 2005:
    case 2009:
    case 2021:
    case 2025:
    case 2033:
    case 2205:
    case 2221:
    case 2229:
    case 2233:
    case 2253:
    case 2261:
    case 2265:
    case 2277:
    case 2281:
    case 2289: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 0 );
        Node &n3 = node( 1 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 1 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 2 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 3 ) );
        Node &no = node( 4 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 5;
        return true;
    }
    case 1285:
    case 1289:
    case 1297:
    case 1317:
    case 1321:
    case 1329:
    case 1349:
    case 1353:
    case 1361:
    case 1381:
    case 1385:
    case 1393:
    case 1413:
    case 1417:
    case 1425:
    case 1445:
    case 1449:
    case 1457:
    case 1477:
    case 1481:
    case 1489:
    case 1509:
    case 1513:
    case 1521: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 0 );
        Node &n3 = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1299:
    case 1331:
    case 1363:
    case 1395:
    case 1427:
    case 1459:
    case 1491:
    case 1523:
    case 1579:
    case 1587:
    case 1643:
    case 1651:
    case 1707:
    case 1715:
    case 1771:
    case 1779:
    case 1883:
    case 1899:
    case 1907:
    case 2011:
    case 2027:
    case 2035:
    case 2235:
    case 2267:
    case 2283:
    case 2291: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 1 );
        Node &n3 = node( 2 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 2 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 3 ) );
        Node &no = node( 3 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 4;
        return true;
    }
    case 1303:
    case 1335:
    case 1367:
    case 1399:
    case 1431:
    case 1463:
    case 1495:
    case 1527:
    case 1591:
    case 1655:
    case 1719:
    case 1783:
    case 1911:
    case 2039:
    case 2295: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 2 );
        Node &n3 = node( 3 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 3 ) );
        Node &no = node( 2 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 3;
        return true;
    }
    case 1584:
    case 1648:
    case 1712:
    case 1776: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 5 );
        Node &n3 = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1840:
    case 1872:
    case 1968:
    case 2000:
    case 2096:
    case 2128:
    case 2192: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 5 );
        Node &n3 = node( 6 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1904:
    case 2032: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 6 );
        Node &n3 = node( 0 );
        Node &nn = node( 5 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 6;
        return true;
    }
    case 2160:
    case 2224:
    case 2256: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 6 );
        Node &n3 = node( 7 );
        Node &nn = node( 5 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 6 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 7;
        return true;
    }
    case 2288: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 7 );
        Node &n3 = node( 0 );
        Node &nn = node( 5 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 6;
        return true;
    }
    case 1283:
    case 1315:
    case 1347:
    case 1379:
    case 1411:
    case 1443:
    case 1475:
    case 1507: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 4 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &n3 = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1287:
    case 1291:
    case 1293:
    case 1319:
    case 1323:
    case 1325:
    case 1351:
    case 1355:
    case 1357:
    case 1383:
    case 1387:
    case 1389:
    case 1415:
    case 1419:
    case 1421:
    case 1447:
    case 1451:
    case 1453:
    case 1479:
    case 1483:
    case 1485:
    case 1511:
    case 1515:
    case 1517: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 4 );
        Node &n1 = node( 0 );
        Node &n2 = node( 2 );
        Node &n3 = node( 3 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 3 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 4 ) );
        size = 4;
        return true;
    }
    case 1295:
    case 1327:
    case 1359:
    case 1391:
    case 1423:
    case 1455:
    case 1487:
    case 1519: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 4 );
        Node &n1 = node( 0 );
        Node &n2 = node( 3 );
        Node &n3 = node( 4 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 4 ) );
        size = 3;
        return true;
    }
    case 1861:
    case 1865:
    case 1873:
    case 1889:
    case 1989:
    case 1993:
    case 2001:
    case 2017:
    case 2189:
    case 2197:
    case 2201:
    case 2213:
    case 2217:
    case 2225:
    case 2245:
    case 2249:
    case 2257:
    case 2273: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 4 );
        Node &n1 = node( 5 );
        Node &n2 = node( 0 );
        Node &n3 = node( 1 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 1 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 2 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 3 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 4 ) );
        Node &no = node( 5 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 6;
        return true;
    }
    case 1541:
    case 1545:
    case 1553:
    case 1569:
    case 1605:
    case 1609:
    case 1617:
    case 1633:
    case 1669:
    case 1673:
    case 1681:
    case 1697:
    case 1733:
    case 1737:
    case 1745:
    case 1761: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 4 );
        Node &n1 = node( 5 );
        Node &n2 = node( 0 );
        Node &n3 = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1571:
    case 1635:
    case 1699:
    case 1763:
    case 1867:
    case 1875:
    case 1891:
    case 1995:
    case 2003:
    case 2019:
    case 2203:
    case 2219:
    case 2227:
    case 2251:
    case 2259:
    case 2275: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 4 );
        Node &n1 = node( 5 );
        Node &n2 = node( 1 );
        Node &n3 = node( 2 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 2 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 3 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 4 ) );
        Node &no = node( 4 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 5;
        return true;
    }
    case 1575:
    case 1639:
    case 1703:
    case 1767:
    case 1879:
    case 1895:
    case 2007:
    case 2023:
    case 2231:
    case 2263:
    case 2279: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 4 );
        Node &n1 = node( 5 );
        Node &n2 = node( 2 );
        Node &n3 = node( 3 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 3 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 4 ) );
        Node &no = node( 3 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 4;
        return true;
    }
    case 1583:
    case 1647:
    case 1711:
    case 1775:
    case 1903:
    case 2031:
    case 2287: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 4 );
        Node &n1 = node( 5 );
        Node &n2 = node( 3 );
        Node &n3 = node( 4 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 4 ) );
        Node &no = node( 2 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 3;
        return true;
    }
    case 1888:
    case 2016: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 4 );
        Node &n1 = node( 5 );
        Node &n2 = node( 6 );
        Node &n3 = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 2144:
    case 2208: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 4 );
        Node &n1 = node( 5 );
        Node &n2 = node( 6 );
        Node &n3 = node( 7 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 2272: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 4 );
        Node &n1 = node( 5 );
        Node &n2 = node( 7 );
        Node &n3 = node( 0 );
        Node &nn = node( 6 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        size = 7;
        return true;
    }
    case 1539:
    case 1603:
    case 1667:
    case 1731: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 5 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &n3 = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1543:
    case 1547:
    case 1549:
    case 1555:
    case 1557:
    case 1561:
    case 1607:
    case 1611:
    case 1613:
    case 1619:
    case 1621:
    case 1625:
    case 1671:
    case 1675:
    case 1677:
    case 1683:
    case 1685:
    case 1689:
    case 1735:
    case 1739:
    case 1741:
    case 1747:
    case 1749:
    case 1753: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 5 );
        Node &n1 = node( 0 );
        Node &n2 = node( 2 );
        Node &n3 = node( 3 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 3 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 5 ) );
        size = 5;
        return true;
    }
    case 1551:
    case 1559:
    case 1563:
    case 1565:
    case 1615:
    case 1623:
    case 1627:
    case 1629:
    case 1679:
    case 1687:
    case 1691:
    case 1693:
    case 1743:
    case 1751:
    case 1755:
    case 1757: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 5 );
        Node &n1 = node( 0 );
        Node &n2 = node( 3 );
        Node &n3 = node( 4 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 5 ) );
        size = 4;
        return true;
    }
    case 1567:
    case 1631:
    case 1695:
    case 1759: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 5 );
        Node &n1 = node( 0 );
        Node &n2 = node( 4 );
        Node &n3 = node( 5 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 5 ) );
        size = 3;
        return true;
    }
    case 2181:
    case 2185:
    case 2193:
    case 2209:
    case 2241: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 5 );
        Node &n1 = node( 6 );
        Node &n2 = node( 0 );
        Node &n3 = node( 1 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 1 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 2 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 3 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 5 ) );
        Node &no = node( 6 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 7;
        return true;
    }
    case 1797:
    case 1801:
    case 1809:
    case 1825:
    case 1857:
    case 1925:
    case 1929:
    case 1937:
    case 1953:
    case 1985: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 5 );
        Node &n1 = node( 6 );
        Node &n2 = node( 0 );
        Node &n3 = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1859:
    case 1987:
    case 2187:
    case 2195:
    case 2211:
    case 2243: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 5 );
        Node &n1 = node( 6 );
        Node &n2 = node( 1 );
        Node &n3 = node( 2 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 2 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 3 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 5 ) );
        Node &no = node( 5 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 6;
        return true;
    }
    case 1863:
    case 1991:
    case 2199:
    case 2215:
    case 2247: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 5 );
        Node &n1 = node( 6 );
        Node &n2 = node( 2 );
        Node &n3 = node( 3 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 3 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 5 ) );
        Node &no = node( 4 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 5;
        return true;
    }
    case 1871:
    case 1999:
    case 2223:
    case 2255: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 5 );
        Node &n1 = node( 6 );
        Node &n2 = node( 3 );
        Node &n3 = node( 4 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 5 ) );
        Node &no = node( 3 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 4;
        return true;
    }
    case 1887:
    case 2015:
    case 2271: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 5 );
        Node &n1 = node( 6 );
        Node &n2 = node( 4 );
        Node &n3 = node( 5 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 5 ) );
        Node &no = node( 2 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 3;
        return true;
    }
    case 2240: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 5 );
        Node &n1 = node( 6 );
        Node &n2 = node( 7 );
        Node &n3 = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1795:
    case 1923: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 6 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &n3 = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 1799:
    case 1803:
    case 1805:
    case 1811:
    case 1813:
    case 1817:
    case 1827:
    case 1829:
    case 1833:
    case 1841:
    case 1927:
    case 1931:
    case 1933:
    case 1939:
    case 1941:
    case 1945:
    case 1955:
    case 1957:
    case 1961:
    case 1969: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 6 );
        Node &n1 = node( 0 );
        Node &n2 = node( 2 );
        Node &n3 = node( 3 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 3 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 6 ) );
        size = 6;
        return true;
    }
    case 1807:
    case 1815:
    case 1819:
    case 1821:
    case 1831:
    case 1835:
    case 1837:
    case 1843:
    case 1845:
    case 1849:
    case 1935:
    case 1943:
    case 1947:
    case 1949:
    case 1959:
    case 1963:
    case 1965:
    case 1971:
    case 1973:
    case 1977: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 6 );
        Node &n1 = node( 0 );
        Node &n2 = node( 3 );
        Node &n3 = node( 4 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 6 ) );
        size = 5;
        return true;
    }
    case 1823:
    case 1839:
    case 1847:
    case 1851:
    case 1853:
    case 1951:
    case 1967:
    case 1975:
    case 1979:
    case 1981: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 6 );
        Node &n1 = node( 0 );
        Node &n2 = node( 4 );
        Node &n3 = node( 5 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 6 ) );
        size = 4;
        return true;
    }
    case 1855:
    case 1983: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 6 );
        Node &n1 = node( 0 );
        Node &n2 = node( 5 );
        Node &n3 = node( 6 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 6 ) );
        size = 3;
        return true;
    }
    case 2053:
    case 2057:
    case 2065:
    case 2081:
    case 2113:
    case 2177: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 6 );
        Node &n1 = node( 7 );
        Node &n2 = node( 0 );
        Node &n3 = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 2179: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 6 );
        Node &n1 = node( 7 );
        Node &n2 = node( 1 );
        Node &n3 = node( 2 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 2 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 3 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 6 ) );
        Node &no = node( 6 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 7;
        return true;
    }
    case 2183: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 6 );
        Node &n1 = node( 7 );
        Node &n2 = node( 2 );
        Node &n3 = node( 3 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 3 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 6 ) );
        Node &no = node( 5 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 6;
        return true;
    }
    case 2191: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 6 );
        Node &n1 = node( 7 );
        Node &n2 = node( 3 );
        Node &n3 = node( 4 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 6 ) );
        Node &no = node( 4 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 5;
        return true;
    }
    case 2207: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 6 );
        Node &n1 = node( 7 );
        Node &n2 = node( 4 );
        Node &n3 = node( 5 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 6 ) );
        Node &no = node( 3 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 4;
        return true;
    }
    case 2239: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 6 );
        Node &n1 = node( 7 );
        Node &n2 = node( 5 );
        Node &n3 = node( 6 );
        Node &nn = node( 0 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n0.dir_x = n2.dir_x; n0.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 1 ).get_straight_content_from( nodes->local_at( 6 ) );
        Node &no = node( 2 );
        if ( store_the_normals ) { no.dir_x = normal.x; no.dir_y = normal.y; }
        no.x = n0.x - m1 * ( n1.x - n0.x );
        no.y = n0.y - m1 * ( n1.y - n0.y );
        no.cut_id.set( cut_id );
        size = 3;
        return true;
    }
    case 2051: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 7 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &n3 = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.cut_id.set( cut_id );
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n2.x = n3.x - m2 * ( n2.x - n3.x );
        n2.y = n3.y - m2 * ( n2.y - n3.y );
        return true;
    }
    case 2055:
    case 2059:
    case 2061:
    case 2067:
    case 2069:
    case 2073:
    case 2083:
    case 2085:
    case 2089:
    case 2097:
    case 2115:
    case 2117:
    case 2121:
    case 2129:
    case 2145: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 7 );
        Node &n1 = node( 0 );
        Node &n2 = node( 2 );
        Node &n3 = node( 3 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 3 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 6 ) );
        nodes->local_at( 6 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 7;
        return true;
    }
    case 2063:
    case 2071:
    case 2075:
    case 2077:
    case 2087:
    case 2091:
    case 2093:
    case 2099:
    case 2101:
    case 2105:
    case 2119:
    case 2123:
    case 2125:
    case 2131:
    case 2133:
    case 2137:
    case 2147:
    case 2149:
    case 2153:
    case 2161: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 7 );
        Node &n1 = node( 0 );
        Node &n2 = node( 3 );
        Node &n3 = node( 4 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 4 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 6 ) );
        nodes->local_at( 5 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 6;
        return true;
    }
    case 2079:
    case 2095:
    case 2103:
    case 2107:
    case 2109:
    case 2127:
    case 2135:
    case 2139:
    case 2141:
    case 2151:
    case 2155:
    case 2157:
    case 2163:
    case 2165:
    case 2169: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 7 );
        Node &n1 = node( 0 );
        Node &n2 = node( 4 );
        Node &n3 = node( 5 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 5 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 6 ) );
        nodes->local_at( 4 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 5;
        return true;
    }
    case 2111:
    case 2143:
    case 2159:
    case 2167:
    case 2171:
    case 2173: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 7 );
        Node &n1 = node( 0 );
        Node &n2 = node( 5 );
        Node &n3 = node( 6 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 6 ) );
        nodes->local_at( 3 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 4;
        return true;
    }
    case 2175: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        Node &n0 = node( 7 );
        Node &n1 = node( 0 );
        Node &n2 = node( 6 );
        Node &n3 = node( 7 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s3 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF m1 = s0 / ( s1 - s0 );
        TF m2 = s3 / ( s2 - s3 );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0.x - m1 * ( n1.x - n0.x );
        n1.y = n0.y - m1 * ( n1.y - n0.y );
        n1.cut_id.set( cut_id );
        if ( store_the_normals ) { nn.dir_x = n2.dir_x; nn.dir_y = n2.dir_y; }
        nn.x = n3.x - m2 * ( n2.x - n3.x );
        nn.y = n3.y - m2 * ( n2.y - n3.y );
        nn.cut_id.set( n2.cut_id.get() );
        nodes->local_at( 2 ).get_straight_content_from( nodes->local_at( 7 ) );
        size = 3;
        return true;
    }
    case 514:
    case 518:
    case 522:
    case 526:
    case 530:
    case 534:
    case 538:
    case 542:
    case 546:
    case 550:
    case 554:
    case 558:
    case 562:
    case 566:
    case 570:
    case 574:
    case 578:
    case 582:
    case 586:
    case 590:
    case 594:
    case 598:
    case 602:
    case 606:
    case 610:
    case 614:
    case 618:
    case 622:
    case 626:
    case 630:
    case 634:
    case 638:
    case 642:
    case 646:
    case 650:
    case 654:
    case 658:
    case 662:
    case 666:
    case 670:
    case 674:
    case 678:
    case 682:
    case 686:
    case 690:
    case 694:
    case 698:
    case 702:
    case 706:
    case 710:
    case 714:
    case 718:
    case 722:
    case 726:
    case 730:
    case 734:
    case 738:
    case 742:
    case 746:
    case 750:
    case 754:
    case 758:
    case 762:
    case 766: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 3;
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 0 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 513:
    case 517:
    case 521:
    case 525:
    case 529:
    case 533:
    case 537:
    case 541:
    case 545:
    case 549:
    case 553:
    case 557:
    case 561:
    case 565:
    case 569:
    case 573:
    case 577:
    case 581:
    case 585:
    case 589:
    case 593:
    case 597:
    case 601:
    case 605:
    case 609:
    case 613:
    case 617:
    case 621:
    case 625:
    case 629:
    case 633:
    case 637:
    case 641:
    case 645:
    case 649:
    case 653:
    case 657:
    case 661:
    case 665:
    case 669:
    case 673:
    case 677:
    case 681:
    case 685:
    case 689:
    case 693:
    case 697:
    case 701:
    case 705:
    case 709:
    case 713:
    case 717:
    case 721:
    case 725:
    case 729:
    case 733:
    case 737:
    case 741:
    case 745:
    case 749:
    case 753:
    case 757:
    case 761:
    case 765: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 3;
        Node &n0 = node( 1 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 2 ).get_straight_content_from( node( 1 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 4;
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 2 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 3 ).get_straight_content_from( node( 2 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 4;
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 0 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 4;
        Node &n0 = node( 2 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 3 ).get_straight_content_from( node( 2 ) );
        node( 2 ).get_straight_content_from( node( 1 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 5;
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 2 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 4 ).get_straight_content_from( node( 3 ) );
        node( 3 ).get_straight_content_from( node( 2 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 5;
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 3 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 4 ).get_straight_content_from( node( 3 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 5;
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 0 );
        Node &nn = node( 4 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
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
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 5;
        Node &n0 = node( 3 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 4 ).get_straight_content_from( node( 3 ) );
        node( 3 ).get_straight_content_from( node( 2 ) );
        node( 2 ).get_straight_content_from( node( 1 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1282:
    case 1314:
    case 1346:
    case 1378:
    case 1410:
    case 1442:
    case 1474:
    case 1506: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 6;
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 2 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 5 ).get_straight_content_from( node( 4 ) );
        node( 4 ).get_straight_content_from( node( 3 ) );
        node( 3 ).get_straight_content_from( node( 2 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1284:
    case 1316:
    case 1348:
    case 1380:
    case 1412:
    case 1444:
    case 1476:
    case 1508: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 6;
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 3 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 5 ).get_straight_content_from( node( 4 ) );
        node( 4 ).get_straight_content_from( node( 3 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1288:
    case 1320:
    case 1352:
    case 1384:
    case 1416:
    case 1448:
    case 1480:
    case 1512: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 6;
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 4 );
        Node &nn = node( 4 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 5 ).get_straight_content_from( node( 4 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1296:
    case 1328:
    case 1360:
    case 1392:
    case 1424:
    case 1456:
    case 1488:
    case 1520: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 6;
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 0 );
        Node &nn = node( 5 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1281:
    case 1313:
    case 1345:
    case 1377:
    case 1409:
    case 1441:
    case 1473:
    case 1505: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 6;
        Node &n0 = node( 4 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 5 ).get_straight_content_from( node( 4 ) );
        node( 4 ).get_straight_content_from( node( 3 ) );
        node( 3 ).get_straight_content_from( node( 2 ) );
        node( 2 ).get_straight_content_from( node( 1 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1538:
    case 1602:
    case 1666:
    case 1730: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 7;
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 2 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 6 ).get_straight_content_from( node( 5 ) );
        node( 5 ).get_straight_content_from( node( 4 ) );
        node( 4 ).get_straight_content_from( node( 3 ) );
        node( 3 ).get_straight_content_from( node( 2 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1540:
    case 1604:
    case 1668:
    case 1732: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 7;
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 3 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 6 ).get_straight_content_from( node( 5 ) );
        node( 5 ).get_straight_content_from( node( 4 ) );
        node( 4 ).get_straight_content_from( node( 3 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1544:
    case 1608:
    case 1672:
    case 1736: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 7;
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 4 );
        Node &nn = node( 4 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 6 ).get_straight_content_from( node( 5 ) );
        node( 5 ).get_straight_content_from( node( 4 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1552:
    case 1616:
    case 1680:
    case 1744: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 7;
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 5 );
        Node &nn = node( 5 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 6 ).get_straight_content_from( node( 5 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1568:
    case 1632:
    case 1696:
    case 1760: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 7;
        Node &n0 = node( 4 );
        Node &n1 = node( 5 );
        Node &n2 = node( 0 );
        Node &nn = node( 6 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1537:
    case 1601:
    case 1665:
    case 1729: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 7;
        Node &n0 = node( 5 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 6 ).get_straight_content_from( node( 5 ) );
        node( 5 ).get_straight_content_from( node( 4 ) );
        node( 4 ).get_straight_content_from( node( 3 ) );
        node( 3 ).get_straight_content_from( node( 2 ) );
        node( 2 ).get_straight_content_from( node( 1 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1794:
    case 1922: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 8;
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 2 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 7 ).get_straight_content_from( node( 6 ) );
        node( 6 ).get_straight_content_from( node( 5 ) );
        node( 5 ).get_straight_content_from( node( 4 ) );
        node( 4 ).get_straight_content_from( node( 3 ) );
        node( 3 ).get_straight_content_from( node( 2 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1796:
    case 1924: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 8;
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 3 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 7 ).get_straight_content_from( node( 6 ) );
        node( 6 ).get_straight_content_from( node( 5 ) );
        node( 5 ).get_straight_content_from( node( 4 ) );
        node( 4 ).get_straight_content_from( node( 3 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1800:
    case 1928: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 8;
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 4 );
        Node &nn = node( 4 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 7 ).get_straight_content_from( node( 6 ) );
        node( 6 ).get_straight_content_from( node( 5 ) );
        node( 5 ).get_straight_content_from( node( 4 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1808:
    case 1936: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 8;
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 5 );
        Node &nn = node( 5 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 7 ).get_straight_content_from( node( 6 ) );
        node( 6 ).get_straight_content_from( node( 5 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1824:
    case 1952: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 8;
        Node &n0 = node( 4 );
        Node &n1 = node( 5 );
        Node &n2 = node( 6 );
        Node &nn = node( 6 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 7 ).get_straight_content_from( node( 6 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1856:
    case 1984: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 8;
        Node &n0 = node( 5 );
        Node &n1 = node( 6 );
        Node &n2 = node( 0 );
        Node &nn = node( 7 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 1793:
    case 1921: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 8;
        Node &n0 = node( 6 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 7 ).get_straight_content_from( node( 6 ) );
        node( 6 ).get_straight_content_from( node( 5 ) );
        node( 5 ).get_straight_content_from( node( 4 ) );
        node( 4 ).get_straight_content_from( node( 3 ) );
        node( 3 ).get_straight_content_from( node( 2 ) );
        node( 2 ).get_straight_content_from( node( 1 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 2050: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 9;
        Node &n0 = node( 0 );
        Node &n1 = node( 1 );
        Node &n2 = node( 2 );
        Node &nn = node( 2 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 8 ).get_straight_content_from( node( 7 ) );
        node( 7 ).get_straight_content_from( node( 6 ) );
        node( 6 ).get_straight_content_from( node( 5 ) );
        node( 5 ).get_straight_content_from( node( 4 ) );
        node( 4 ).get_straight_content_from( node( 3 ) );
        node( 3 ).get_straight_content_from( node( 2 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 2052: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 9;
        Node &n0 = node( 1 );
        Node &n1 = node( 2 );
        Node &n2 = node( 3 );
        Node &nn = node( 3 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 8 ).get_straight_content_from( node( 7 ) );
        node( 7 ).get_straight_content_from( node( 6 ) );
        node( 6 ).get_straight_content_from( node( 5 ) );
        node( 5 ).get_straight_content_from( node( 4 ) );
        node( 4 ).get_straight_content_from( node( 3 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 2056: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 9;
        Node &n0 = node( 2 );
        Node &n1 = node( 3 );
        Node &n2 = node( 4 );
        Node &nn = node( 4 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 2 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 8 ).get_straight_content_from( node( 7 ) );
        node( 7 ).get_straight_content_from( node( 6 ) );
        node( 6 ).get_straight_content_from( node( 5 ) );
        node( 5 ).get_straight_content_from( node( 4 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 2064: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 9;
        Node &n0 = node( 3 );
        Node &n1 = node( 4 );
        Node &n2 = node( 5 );
        Node &nn = node( 5 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 3 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 8 ).get_straight_content_from( node( 7 ) );
        node( 7 ).get_straight_content_from( node( 6 ) );
        node( 6 ).get_straight_content_from( node( 5 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 2080: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 9;
        Node &n0 = node( 4 );
        Node &n1 = node( 5 );
        Node &n2 = node( 6 );
        Node &nn = node( 6 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 4 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 8 ).get_straight_content_from( node( 7 ) );
        node( 7 ).get_straight_content_from( node( 6 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 2112: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 9;
        Node &n0 = node( 5 );
        Node &n1 = node( 6 );
        Node &n2 = node( 7 );
        Node &nn = node( 7 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 5 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 8 ).get_straight_content_from( node( 7 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 2176: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 9;
        Node &n0 = node( 6 );
        Node &n1 = node( 7 );
        Node &n2 = node( 0 );
        Node &nn = node( 8 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 6 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
    }
    case 2049: {
        if ( store_the_normals && ( flags & ConvexPolyhedron::plane_cut_flag_dir_is_normalized ) == 0 )
            normal /= norm_2( normal );
        size = 9;
        Node &n0 = node( 7 );
        Node &n1 = node( 0 );
        Node &n2 = node( 1 );
        Node &nn = node( 1 );
        TF s0 = reinterpret_cast<const TF *>( &di_0 )[ 7 ];
        TF s1 = reinterpret_cast<const TF *>( &di_0 )[ 0 ];
        TF s2 = reinterpret_cast<const TF *>( &di_0 )[ 1 ];
        TF m0 = s0 / ( s1 - s0 );
        TF m1 = s2 / ( s1 - s2 );
        TF n0_x = n0.x;
        TF n0_y = n0.y;
        node( 8 ).get_straight_content_from( node( 7 ) );
        node( 7 ).get_straight_content_from( node( 6 ) );
        node( 6 ).get_straight_content_from( node( 5 ) );
        node( 5 ).get_straight_content_from( node( 4 ) );
        node( 4 ).get_straight_content_from( node( 3 ) );
        node( 3 ).get_straight_content_from( node( 2 ) );
        node( 2 ).get_straight_content_from( node( 1 ) );
        if ( store_the_normals ) { nn.dir_x = n1.dir_x; nn.dir_y = n1.dir_y; }
        nn.x = n2.x - m1 * ( n1.x - n2.x );
        nn.y = n2.y - m1 * ( n1.y - n2.y );
        nn.cut_id.set( n1.cut_id.get() );
        if ( store_the_normals ) { n1.dir_x = normal.x; n1.dir_y = normal.y; }
        n1.x = n0_x - m0 * ( n1.x - n0_x );
        n1.y = n0_y - m0 * ( n1.y - n0_y );
        n1.cut_id.set( cut_id );
        return true;
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
        return false;
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
    case 515:
    case 519:
    case 523:
    case 527:
    case 531:
    case 535:
    case 539:
    case 543:
    case 547:
    case 551:
    case 555:
    case 559:
    case 563:
    case 567:
    case 571:
    case 575:
    case 579:
    case 583:
    case 587:
    case 591:
    case 595:
    case 599:
    case 603:
    case 607:
    case 611:
    case 615:
    case 619:
    case 623:
    case 627:
    case 631:
    case 635:
    case 639:
    case 643:
    case 647:
    case 651:
    case 655:
    case 659:
    case 663:
    case 667:
    case 671:
    case 675:
    case 679:
    case 683:
    case 687:
    case 691:
    case 695:
    case 699:
    case 703:
    case 707:
    case 711:
    case 715:
    case 719:
    case 723:
    case 727:
    case 731:
    case 735:
    case 739:
    case 743:
    case 747:
    case 751:
    case 755:
    case 759:
    case 763:
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
        return true;
    }    default: break;
    }
    #endif // __AVX512F__
    return plane_cut_gen( origin, normal, cut_id, N<flags>() );
}

template<class Pc> template<int flags,class T>
bool ConvexPolyhedron2<Pc>::plane_cut_simd_switch( Pt origin, Pt normal, CI cut_id, N<flags>, S<T> ) {
    return plane_cut_gen( origin, normal, cut_id, N<flags>() );
}

} // namespace sdot
