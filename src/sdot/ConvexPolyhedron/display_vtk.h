#ifndef DISPLAY_VTK_H
#define DISPLAY_VTK_H

//template<class Pc>
//void ConvexPolyhedron<Pc,void>::display_vtk( VtkOutput &vo, const std::vector<TF> &cell_values, Pt offset, bool /*display_both_sides*/ ) const {
//    std::vector<VtkOutput::Pt> pts;
//    pts.reserve( 16 );
//    for_each_face( [&]( const Face &face ) {
//        pts.clear();
//        for( int n : face.nodes )
//            pts.push_back( nodes[ n ].pos() + offset );
//        vo.add_polygon( pts, cell_values );
//    } );
//}


#endif // DISPLAY_VTK_H
