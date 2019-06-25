#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"

//// nsmake lib_path /usr/local/lib
//// nsmake lib_name geogram
//// nsmake cpp_flag -O3

#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/stopwatch.h>
#include <geogram/basic/file_system.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_reorder.h>
#include <geogram/delaunay/delaunay.h>
#include <algorithm>

//namespace {
//    using namespace GEO;

//   /**
//    * \brief Loads points from a file.
//    * \param[in] points_filename the name of the file with the points.
//    *  -If the example was compiled with the Geogram library, then any
//    *  mesh file handled by Geogram can be used.
//    *  -if the example was compiled with Delaunay_psm (single file), then
//    *  the file should be ASCII, with one point per line.
//    * \param[in] dimension number of coordinates of the points.
//    * \param[out] points the loaded points, in a single vector of coordinates.
//    *  In the end, the number of loaded points is points.size()/dimension.
//    */
//    bool load_points(
//    const std::string& points_filename,
//    index_t dimension,
//    vector<double>& points
//    ) {
//#ifdef GEOGRAM_PSM
//    // Simple data input: one point per line, coordinates in ASCII
//    LineInput input(points_filename);
//    if(!input.OK()) {
//        return false;
//    }
//    while(!input.eof() && input.get_line()) {
//        input.get_fields();
//        if(input.nb_fields() == dimension) {
//        for(index_t c=0; c<dimension; ++c) {
//            points.push_back(input.field_as_double(c));
//        }
//        }
//    }
//#else
//    // Using Geogram mesh I/O
//    Mesh M;
//    MeshIOFlags flags;
//    flags.reset_element(MESH_FACETS);
//    flags.reset_element(MESH_CELLS);
//    if(!mesh_load(points_filename, M, flags)) {
//        return false;
//    }
//    M.vertices.set_dimension(dimension);
//    index_t nb_points = M.vertices.nb();
//    points.resize(nb_points * dimension);
//    Memory::copy(
//        points.data(),
//        M.vertices.point_ptr(0),
//        M.vertices.nb()*dimension*sizeof(double)
//    );
//#endif
//    return true;
//    }

//   /**
//    * \brief Saves a Delaunay triangulation to a file.
//    * \param[in] delaunay a pointer to the Delaunay triangulation.
//    * \param[in] filename the name of the file to be saved.
//    *  -If the example was compiled with the Geogram library, then any
//    *  mesh file handled by Geogram can be used.
//    *  if the example was compiled with Delaunay_psm (single file), then
//    *  the points and vertices of the triangulation are output in ASCII.
//    * \param[in] convex_hull_only if true, then only the triangles on the
//    *  convex hull are output.
//    */
//    void save_Delaunay(
//    Delaunay* delaunay, const std::string& filename,
//    bool convex_hull_only = false
//    ) {
//    vector<index_t> tri2v;

//    if(convex_hull_only) {

//        // The convex hull can be efficiently traversed only if infinite
//        // tetrahedra are kept.
//        geo_assert(delaunay->keeps_infinite());

//        // The convex hull can be retrieved as the finite facets
//        // of the infinite cells (note: it would be also possible to
//        // throw away the infinite cells and get the convex hull as
//        // the facets adjacent to no cell). Here we use the infinite
//        // cells to show an example with them.


//        // This block is just a sanity check
//        {
//        for(index_t t=0; t < delaunay->nb_finite_cells(); ++t) {
//            geo_debug_assert(delaunay->cell_is_finite(t));
//        }

//        for(index_t t=delaunay->nb_finite_cells();
//            t < delaunay->nb_cells(); ++t) {
//            geo_debug_assert(delaunay->cell_is_infinite(t));
//        }
//        }

//        // This iterates on the infinite cells
//        for(
//        index_t t = delaunay->nb_finite_cells();
//        t < delaunay->nb_cells(); ++t
//         ) {
//        for(index_t lv=0; lv<4; ++lv) {
//            signed_index_t v = delaunay->cell_vertex(t,lv);
//            if(v != -1) {
//            tri2v.push_back(index_t(v));
//            }
//        }
//        }
//    }

//#ifdef GEOGRAM_PSM
//    // Simple data output: output vertices and simplices

//    Logger::out("Delaunay") << "Saving output to " << filename << std::endl;
//    std::ofstream out(filename.c_str());

//    out << delaunay->nb_vertices() << " vertices" << std::endl;
//    for(index_t v=0; v < delaunay->nb_vertices(); ++v) {
//        for(index_t c=0; c < delaunay->dimension(); ++c) {
//        out << delaunay->vertex_ptr(v)[c] << " ";
//        }
//        out << std::endl;
//    }
//    if(convex_hull_only) {
//        out << tri2v.size()/3 << " simplices" << std::endl;
//        for(index_t t=0; t<tri2v.size()/3; ++t) {
//        out << tri2v[3*t] << " "
//            << tri2v[3*t+1] << " "
//            << tri2v[3*t+2] << std::endl;
//        }
//    } else {
//        out << delaunay->nb_cells() << " simplices" << std::endl;
//        for(index_t t=0; t<delaunay->nb_cells(); ++t) {
//        for(index_t lv=0; lv<delaunay->cell_size(); ++lv) {
//            out << delaunay->cell_vertex(t,lv) << " ";
//        }
//        out << std::endl;
//        }
//    }

//#else
//    // Using Geogram mesh I/O: copy Delaunay into a Geogram
//    // mesh and save it to disk.

//    Mesh M_out;
//    vector<double> pts(delaunay->nb_vertices() * 3);
//    for(index_t v = 0; v < delaunay->nb_vertices(); ++v) {
//        pts[3 * v] = delaunay->vertex_ptr(v)[0];
//        pts[3 * v + 1] = delaunay->vertex_ptr(v)[1];
//        pts[3 * v + 2] =
//        (delaunay->dimension() >= 3) ? delaunay->vertex_ptr(v)[2] : 0.0;
//    }

//    if(convex_hull_only) {
//        M_out.facets.assign_triangle_mesh(3, pts, tri2v, true);
//    } else if(delaunay->dimension() == 3) {
//        vector<index_t> tet2v(delaunay->nb_cells() * 4);
//        for(index_t t = 0; t < delaunay->nb_cells(); ++t) {
//        tet2v[4 * t] = index_t(delaunay->cell_vertex(t, 0));
//        tet2v[4 * t + 1] = index_t(delaunay->cell_vertex(t, 1));
//        tet2v[4 * t + 2] = index_t(delaunay->cell_vertex(t, 2));
//        tet2v[4 * t + 3] = index_t(delaunay->cell_vertex(t, 3));
//        }
//        M_out.cells.assign_tet_mesh(3, pts, tet2v, true);
//    } else if(delaunay->dimension() == 2) {
//        tri2v.resize(delaunay->nb_cells() * 3);
//        for(index_t t = 0; t < delaunay->nb_cells(); ++t) {
//        tri2v[3 * t] = index_t(delaunay->cell_vertex(t, 0));
//        tri2v[3 * t + 1] = index_t(delaunay->cell_vertex(t, 1));
//        tri2v[3 * t + 2] = index_t(delaunay->cell_vertex(t, 2));
//        }
//        M_out.facets.assign_triangle_mesh(3, pts, tri2v, true);
//    }
//    M_out.show_stats();

//    Logger::div("Saving the result");
//    MeshIOFlags flags;
//    flags.set_element(MESH_FACETS);
//    flags.set_element(MESH_CELLS);
//    mesh_save(M_out, filename, flags);
//#endif
//    }

//}
////    std::string del = "default"; // CmdLine::get_arg("algo:delaunay");
////    if ( del == "default" ) {
////    }

int main( int /*argc*/, char** /*argv*/ ) {
    using namespace GEO;

    // Needs to be called once.
    GEO::initialize();

    int dimension = 2;

    if ( dimension == 3 ) {
        if( DelaunayFactory::has_creator( "PDEL" ) )
            CmdLine::set_arg("algo:delaunay", "PDEL"); // PDEL = Parallel 3D Delaunay
        else
            CmdLine::set_arg("algo:delaunay", "BDEL"); // BDEL = Sequential 3D Delaunay
    } else if( dimension == 2 )
        CmdLine::set_arg("algo:delaunay", "BDEL2d"); // BDEL2d = Sequential 2D Delaunay

    Delaunay_var delaunay = Delaunay::create( coord_index_t( dimension ) );

    vector<double> points;
    for( std::size_t i = 0; i < 10000000; ++i ) {
        double x = double( rand() ) / RAND_MAX;
        double y = double( rand() ) / RAND_MAX;
        points.push_back( x );
        points.push_back( y );
        //        TF x = double( rand() ) / RAND_MAX;
        //        TF y = double( rand() ) / RAND_MAX;
        //        positions.push_back( { 0.0 + 0.05 * x + 0.10 * y, y } );
        //        positions.push_back( { 1.0 - 0.05 * x - 0.35 * y, y } );
        //        weights.push_back( 0.0 );
        //        weights.push_back( 0.0 );
    }
    index_t nb_points = points.size() / dimension;

    std::uint64_t t0_grid = 0, t1_grid = 0;
    RDTSC_START( t0_grid );
    delaunay->set_vertices( nb_points, points.data() );

    index_t s = 0;
    for( index_t t = 0; t < delaunay->nb_cells(); ++t )
        for( index_t lv = 0; lv < delaunay->cell_size(); ++lv )
            s += delaunay->cell_vertex( t, lv );

    RDTSC_FINAL( t1_grid );

    P( delaunay->nb_cells(), s, ( t1_grid - t0_grid ) / nb_points );
}

