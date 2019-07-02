//// nsmake avoid_inc CGAL/

//// nsmake cpp_flag -march=native
//// nsmake cpp_flag -O5
//// nsmake lib_flag -O5

//// nsmake lib_path /usr/local/lib
//// nsmake lib_name CGAL
//// nsmake lib_name gmp

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <fstream>
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_triangulation_2<K>  Triangulation;
typedef Triangulation::Edge_iterator  Edge_iterator;
typedef Triangulation::Point          Point;

int main() {
    //std::ifstream in("data/voronoi.cin");
    //std::istream_iterator<Point> begin(in);
    //std::istream_iterator<Point> end;
    std::vector<Point> points;
    for( size_t i = 0; i < 100000; i++ ) {
        double x = double( rand() ) / RAND_MAX;
        double y = double( rand() ) / RAND_MAX;
        points.emplace_back( x, y );
    }
    std::cout << points[ 0 ] << std::endl;
    
    Triangulation T;
    T.insert( points.begin(), points.end() );

    int ns = 0, nr = 0;
    Edge_iterator eit = T.edges_begin();
    for ( ; eit !=T.edges_end(); ++eit) {
        CGAL::Object o = T.dual(eit);
        if (CGAL::object_cast<K::Segment_2>(&o)) {++ns;}
        else if (CGAL::object_cast<K::Ray_2>(&o)) {++nr;}
    }
    std::cout << "The Voronoi diagram has " << ns << " finite edges " << " and " << nr << " rays" << std::endl;

    return 0;
}
