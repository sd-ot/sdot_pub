#include "../../src/sdot/Support/Time.h"
#include "../../src/sdot/Support/P.h"
#include <cnpy.h>

//// nsmake lib_path /usr/local/lib
//// nsmake lib_name geogram

// // nsmake cpp_flag -march=native
//// nsmake cpp_flag -O3
//// nsmake lib_name z

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
#include <geogram/voronoi/RVD_callback.h>
#include <geogram/voronoi/RVD.h>
#include <geogram/voronoi/CVT.h>
#include <algorithm>

using namespace GEO;

class SaveRVDCells : public RVDPolyhedronCallback {
public:

    /**
     * \brief SaveRVDCells constructor.
     * \param[out] output_mesh a reference to the generated mesh
     */
    SaveRVDCells() {
        my_vertex_map_ = nullptr;

        // If set, then only one polyhedron per (connected component of) restricted Voronoi
        // cell is generated.
        set_simplify_internal_tet_facets(true);

        // If set, then only one polygon per Voronoi facet is generated.
        set_simplify_voronoi_facets(true);

        // If set, then the intersection between a Voronoi cell and the boundary surface is
        // replaced with a single polygon whenever possible (i.e. when its topology is a
        // disk and when it has at least 3 corners).
        set_simplify_boundary_facets(true);
    }

    ~SaveRVDCells() {
    }

    virtual void begin() {
    }
    virtual void end() {
    }

    virtual void begin_polyhedron( index_t /*seed*/, index_t /*tetrahedron*/) {
        ++s;
    }


    virtual void begin_facet(index_t /*facet_seed*/, index_t facet_tet) {
        ++s;
        // P( facet_tet );
    }

    virtual void vertex( const double* /*geometry*/, const GEOGen::SymbolicVertex& /*symb*/ ) {
        ++s;
    }

    virtual void end_facet() {
    }

    virtual void end_polyhedron() {
    }

    virtual void process_polyhedron_mesh() {
    }

    double s = 0;
private:
    vector<index_t> current_facet_;
    RVDVertexMap* my_vertex_map_;
};

int main( int /*argc*/, char** /*argv*/ ) {
    constexpr int dim = 3;
    using TF = double;

    // Needs to be called once.
    GEO::initialize();

    CmdLine::import_arg_group( "standard" );
    CmdLine::import_arg_group( "algo" );


    CmdLine::declare_arg("RDT", false, "save RDT");
    CmdLine::declare_arg("RVD", true, "save RVD");
    CmdLine::declare_arg("RVD_cells", false, "use new API for computing RVD cells (implies volumetric)");
    CmdLine::declare_arg_group("RVD_cells", "RVD cells simplification flags");
    CmdLine::declare_arg("RVD_cells:simplify_tets", true, "Simplify tets intersections");
    CmdLine::declare_arg("RVD_cells:simplify_voronoi", true, "Simplify Voronoi facets");
    CmdLine::declare_arg("RVD_cells:simplify_boundary", false, "Simplify boundary facets");
    CmdLine::declare_arg("RVD_cells:shrink", 0.0, "Shrink factor for computed cells");

    CmdLine::set_arg("volumetric",true);

    //    Logger::instance()->register_client( new ConsoleLogger );
    //    Logger::instance()->set_quiet( false );

    if ( dim == 3 ) {
        if( DelaunayFactory::has_creator( "PDEL" ) )
            CmdLine::set_arg("algo:delaunay", "PDEL"); // PDEL = Parallel 3D Delaunay
        else
            CmdLine::set_arg("algo:delaunay", "BDEL"); // BDEL = Sequential 3D Delaunay
    } else if( dim == 2 )
        CmdLine::set_arg("algo:delaunay", "BDEL2d"); // BDEL2d = Sequential 2D Delaunay

    //    const char *filenames[] = {
    //        "/data/sdot/faces_20p_2D_1000000.txt",
    //        "/data/sdot/faces_20p_2D_1600000.txt",
    //        "/data/sdot/faces_20p_2D_2560000.txt",
    //        "/data/sdot/faces_20p_2D_4096000.txt",
    //        "/data/sdot/faces_20p_2D_6553600.txt",
    //        "/data/sdot/faces_20p_2D_10485760.txt",
    //        "/data/sdot/faces_20p_2D_16777216.txt",
    //        "/data/sdot/faces_20p_2D_26843545.txt",
    //        //        "/data/sdot/faces_20p_2D_42949672.txt",
    //        //        "/data/sdot/faces_20p_2D_68719475.txt",
    //        //        "/data/sdot/faces_20p_2D_109951160.txt",
    //        //        "/data/sdot/faces_20p_2D_175921856.txt",
    //        //        "/data/sdot/faces_20p_2D_281474969.txt",
    //    };
    const char *filenames[] = {
        //        "/data/sdot/uniform_100000_2D_solved.npy",
        //        "/data/sdot/uniform_200000_2D_solved.npy",
        //        "/data/sdot/uniform_400000_2D_solved.npy",
        //        "/data/sdot/uniform_800000_2D_solved.npy",
        //        "/data/sdot/uniform_1600000_2D_solved.npy",
        "/data/sdot/uniform_12500_3D_solved.npy",
        "/data/sdot/uniform_25000_3D_solved.npy",
        "/data/sdot/uniform_50000_3D_solved.npy",
        "/data/sdot/uniform_100000_3D_solved.npy",
        "/data/sdot/uniform_200000_3D_solved.npy",
    };
    bool use_npy = true;

    // Logger::instance()->set_quiet( false );

    std::cout << "    \\addplot coordinates {\n";
    for( const char *filename : filenames ) {
        std::size_t nb_diracs = 0;
        vector<double> points;

        // read
        if ( use_npy ) {
            cnpy::NpyArray arr = cnpy::npy_load( filename );
            nb_diracs = arr.shape[ 1 ];
            double *data = arr.data<double>();

            for( std::size_t n = 0; n < nb_diracs; ++n ) {
                for( std::size_t d = 0; d < dim; ++d )
                    points.push_back( data[ n + d * nb_diracs ] );
                points.push_back( data[ n + dim * nb_diracs ] );
            }
        } else {
            std::ifstream fin( filename );
            while ( true ) {
                TF p[ dim ], w;
                for( std::size_t d = 0; d < dim; ++d )
                    fin >> p[ d ];
                fin >> w;
                if ( ! fin )
                    break;
                for( std::size_t d = 0; d < dim; ++d )
                    points.push_back( p[ d ] );
                ++nb_diracs;
            }
        }

        //
        Mesh M_in;
        MeshIOFlags flags;
        flags.set_element( MESH_CELLS );
        // mesh_load( "tests/benchmarks/triangle_geogram.mesh", M_in, flags );
        mesh_load( "tests/benchmarks/tetra_geogram.mesh", M_in, flags );

        std::uint64_t t0_grid = 0, t1_grid = 0, nb_reps = 50;
        SaveRVDCells callback;

        RDTSC_START( t0_grid );
        for( std::size_t rep = 0; rep < nb_reps; ++rep ) {
            Mesh surface;
            RegularWeightedDelaunay3d *delaunay = new RegularWeightedDelaunay3d();
            // Delaunay_var delaunay = Delaunay::create( coord_index_t( dim ) );
            delaunay->set_stores_neighbors( true );
            delaunay->set_stores_cicl( true );

            RestrictedVoronoiDiagram_var RVD =  RestrictedVoronoiDiagram::create( delaunay, &M_in );
            RVD->set_volumetric( true );

            delaunay->set_vertices( nb_diracs, points.data() );

            RestrictedVoronoiDiagram::RDTMode mode = RestrictedVoronoiDiagram::RDTMode( RestrictedVoronoiDiagram::RDT_MULTINERVE | RestrictedVoronoiDiagram::RDT_RVC_CENTROIDS  );
            RVD->compute_RDT( surface, mode );

            RVD->for_each_polyhedron( callback );

            auto cell_size = delaunay->cell_size();
            for( std::size_t nc = 0; nc < delaunay->nb_cells(); ++nc ) {
                for( std::size_t i = 0; i < cell_size; ++i ) {
                    auto c = delaunay->cell_adjacent( nc, i );
                    if ( c >= 0 )
                        for( std::size_t si = 0; si < cell_size; ++si )
                            callback.s += points[ delaunay->cell_vertex( c, si ) ];
                }
            }
        }

        RDTSC_FINAL( t1_grid );

        std::cout << "       ( " << nb_diracs << ", " << ( t1_grid - t0_grid ) / ( nb_diracs * nb_reps ) << " ) % " << callback.s << "\n";
    }
    std::cout << "    };\n";
}

