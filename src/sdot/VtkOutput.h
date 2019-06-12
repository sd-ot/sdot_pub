#pragma once

#include "../Geometry/Point2.h"
#include <vector>
#include <mutex>
#include <deque>

namespace sdot {

/**
  Class for simplified (and not optimized) vtk output
*/
class VtkOutput {
public:
    using                    TF                = double;
    using                    Pt                = Point3<TF>;

    /**/                     VtkOutput         ( const std::vector<std::string> &cell_fields_names = {}, const std::vector<std::string> &point_fields_names = {} )

    void                     save              ( std::string filename ) const;
    void                     save              ( std::ostream &os ) const;

    void                     add_arc           ( Pt center, PT A, PT B, PT tangent, const CV &cell_value = {}, unsigned n = 50 );
    void                     add_point         ( Pt p, const CV &cell_value = {} );
    void                     add_lines         ( const std::vector<PT> &p, const CV &cell_value = {} );
    void                     add_lines         ( const std::vector<P2> &p, const CV &cell_value = {} );
    void                     add_arrow         ( PT center, PT dir, const CV &cell_value = {} );
    void                     add_circle        ( PT center, PT normal, TF radius, const CV &cell_value = {}, unsigned n = 50 );
    void                     add_polygon       ( const std::vector<PT> &p, const CV &cell_value = {} );

    std::mutex               mutex;

private:
    struct                   Polygon           { std::vector<Pt> p; };
    struct                   Field             { std::vector<TF> v; };
    struct                   Point             { Pt              p; };
    struct                   Line              { std::vector<Pt> p; };

    std::vector<Field>       point_field_values;
    std::vector<Field>       cell_field_values;
    std::vector<std::string> point_fields_names;
    std::vector<std::string> cell_fields_names;
    std::vector<Polygon>     polygons;
    std::deque<Point>        points;
    std::deque<Line>         lines;
};

} // namespace sdot
