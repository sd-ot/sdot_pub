#include "display_vtk.h"

namespace sdot {

template<class Pc,class Opt>
void display_vtk( VtkOutput &vo, const ConvexPolyhedron<Pc,2,Opt> &cp, const DisplayVtkConvexPolyhedronParams &params ) {
    std::vector<VtkOutput::Pt> pts;
    pts.resize( cp.nb_nodes() );
    for( std::size_t i = 0; i < cp.nb_nodes(); ++i )
        pts[ i ] = VtkOutput::Pt{ cp.node( i )[ 0 ], cp.node( i )[ 1 ], 0 } + params.offset;
    vo.add_polygon( pts, params.cell_values );
}

template<class Pc,class Opt>
void display_vtk( VtkOutput &vo, const ConvexPolyhedron<Pc,3,Opt> &cp, const DisplayVtkConvexPolyhedronParams &params ) {
    std::vector<VtkOutput::Pt> pts;
    pts.reserve( 16 );
    cp.for_each_boundary_item( [&]( const auto &face ) {
        pts.clear();
        face.for_each_node( [&]( auto p ) {
            pts.push_back( p + params.offset );
        } );
        vo.add_polygon( pts, params.cell_values );
    } );

}

} // namespace sdot
