//// nsmake avoid_inc CGAL/

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O5
//// nsmake lib_flag -O5

//// nsmake lib_path /usr/local/lib
//// nsmake lib_name CGAL
//// nsmake lib_name gmp

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include "../../src/sdot/Support/Time.h"
#include <iostream>
#include <fstream>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_3<K> Triangulation;
typedef Triangulation::Facet_iterator Facet_iterator;
typedef Triangulation::Point Point;

void bench( const char *filename ) {
    std::vector<Point> points;
    std::ifstream fin( filename );
    while ( true ) {
        double x, y, z, w;
        fin >> x >> y >> z >> w;
        if ( ! fin )
            break;
        points.push_back( { x, y, z } );
    }
    // std::cout << filename << " " << points.size() << std::endl;

    std::uint64_t t0 = 0, t1 = 0, tm = 0;
    RDTSC_START( t0 );
    Triangulation T;
    T.insert( points.begin(), points.end() );

    int ns = 0;

    RDTSC_START( tm );
    auto eik = T.finite_cells_begin();
    for ( ; eik != T.finite_cells_end(); ++eik ) {
        ns += eik->is_valid();
    }
    RDTSC_FINAL( t1 );

    std::cout << ns << " nb_points: " << points.size() << " cycles: " << ( t1 - t0 ) / points.size() << std::endl;
}

int main() {
    const char *filenames[] = {
        "/data/sdot/faces_20p_3D_1000000.txt",
        "/data/sdot/faces_20p_3D_1600000.txt",
        "/data/sdot/faces_20p_3D_2560000.txt",
        "/data/sdot/faces_20p_3D_4096000.txt",
        "/data/sdot/faces_20p_3D_6553600.txt",
        "/data/sdot/faces_20p_3D_10485760.txt",
        "/data/sdot/faces_20p_3D_16777216.txt",
        "/data/sdot/faces_20p_3D_26843545.txt",
        "/data/sdot/faces_20p_3D_42949672.txt",
        "/data/sdot/faces_20p_3D_68719475.txt",
        "/data/sdot/faces_20p_3D_109951160.txt",
        "/data/sdot/faces_20p_3D_175921856.txt",
        "/data/sdot/faces_20p_3D_281474969.txt",
    };

    for( const char *f : filenames )
        bench( f );
    return 0;
}
