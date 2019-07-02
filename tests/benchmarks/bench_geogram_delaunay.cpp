#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"

//// nsmake lib_path /usr/local/lib
//// nsmake lib_name geogram

//// nsmake cpp_flag -march=native
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
#include <geogram/delaunay/delaunay_2d.h>
#include <geogram/delaunay/delaunay_3d.h>
#include <algorithm>

using namespace GEO;

/**
    * \brief Saves a Delaunay triangulation to a file.
    * \param[in] delaunay a pointer to the Delaunay triangulation.
    * \param[in] filename the name of the file to be saved.
    *  -If the example was compiled with the Geogram library, then any
    *  mesh file handled by Geogram can be used.
    *  if the example was compiled with Delaunay_psm (single file), then
    *  the points and vertices of the triangulation are output in ASCII.
    * \param[in] convex_hull_only if true, then only the triangles on the
    *  convex hull are output.
    */
void save_Delaunay( Delaunay* delaunay, const std::string& filename ) {
    vector<index_t> tri2v;

    // Using Geogram mesh I/O: copy Delaunay into a Geogram
    // mesh and save it to disk.

    Mesh M_out;
    vector<double> pts(delaunay->nb_vertices() * 3);
    for(index_t v = 0; v < delaunay->nb_vertices(); ++v) {
        pts[3 * v + 0] = delaunay->vertex_ptr(v)[0];
        pts[3 * v + 1] = delaunay->vertex_ptr(v)[1];
        pts[3 * v + 2] = (delaunay->dimension() >= 3) ? delaunay->vertex_ptr(v)[2] : 0.0;
    }

    if(delaunay->dimension() == 3) {
        vector<index_t> tet2v(delaunay->nb_cells() * 4);
        for(index_t t = 0; t < delaunay->nb_cells(); ++t) {
            tet2v[4 * t] = index_t(delaunay->cell_vertex(t, 0));
            tet2v[4 * t + 1] = index_t(delaunay->cell_vertex(t, 1));
            tet2v[4 * t + 2] = index_t(delaunay->cell_vertex(t, 2));
            tet2v[4 * t + 3] = index_t(delaunay->cell_vertex(t, 3));
        }
        M_out.cells.assign_tet_mesh(3, pts, tet2v, true);
    } else if(delaunay->dimension() == 2) {
        tri2v.resize(delaunay->nb_cells() * 3);
        for(index_t t = 0; t < delaunay->nb_cells(); ++t) {
            tri2v[3 * t] = index_t(delaunay->cell_vertex(t, 0));
            tri2v[3 * t + 1] = index_t(delaunay->cell_vertex(t, 1));
            tri2v[3 * t + 2] = index_t(delaunay->cell_vertex(t, 2));
        }
        M_out.facets.assign_triangle_mesh(3, pts, tri2v, true);
    }
    M_out.show_stats();

    Logger::div("Saving the result");
    MeshIOFlags flags;
    flags.set_element(MESH_FACETS);
    flags.set_element(MESH_CELLS);
    mesh_save(M_out, filename, flags);
}

int main( int /*argc*/, char** /*argv*/ ) {
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
    // CmdLine::set_arg("algo:delaunay", "BDEL2d"); // BDEL2d = Sequential 2D Delaunay // BPOW2d => puissance

    std::cout << "    \\addplot coordinates {\n";
    for( std::size_t nb_diracs : { 1000000, 2000000, 4000000, 8000000, 16000000, 32000000, 64000000, 128000000 } ) {
        // for( std::size_t nb_diracs : { 100 } ) {
        Delaunay_var delaunay = Delaunay::create( coord_index_t( dimension ) );
        // RegularWeightedDelaunay2d *delaunay = new RegularWeightedDelaunay2d( dimension );


        vector<double> points;
        for( std::size_t i = 0; i < nb_diracs; ++i ) {
            double x = double( rand() ) / RAND_MAX;
            double y = double( rand() ) / RAND_MAX;
            // double w = 0;
            points.push_back( x );
            points.push_back( y );
            // points.push_back( w );
            //        TF x = double( rand() ) / RAND_MAX;
            //        TF y = double( rand() ) / RAND_MAX;
            //        positions.push_back( { 0.0 + 0.05 * x + 0.10 * y, y } );
            //        positions.push_back( { 1.0 - 0.05 * x - 0.35 * y, y } );
            //        weights.push_back( 0.0 );
            //        weights.push_back( 0.0 );
        }
        std::uint64_t t0_grid = 0, t1_grid = 0;
        RDTSC_START( t0_grid );
        delaunay->set_vertices( nb_diracs, points.data() );

        //        for( index_t t = 0; t < delaunay->nb_cells(); ++t ) {
        //            P( t );
        //            for( index_t lv = 0; lv < delaunay->cell_size(); ++lv )
        //                P( delaunay->cell_vertex( t, lv ) );
        //        }

        index_t s = 0;
        for( index_t t = 0; t < delaunay->nb_cells(); ++t )
            for( index_t lv = 0; lv < delaunay->cell_size(); ++lv )
                s += delaunay->cell_vertex( t, lv );
        RDTSC_FINAL( t1_grid );

        // save_Delaunay( delaunay, "proute.mesh" );

        std::cout << "       ( " << nb_diracs << ", " << ( t1_grid - t0_grid ) / nb_diracs << " ) % " << s << "\n";
    }
    std::cout << "    };\n";
}

