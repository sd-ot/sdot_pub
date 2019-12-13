#ifndef DISPLAY_VTK_H
#define DISPLAY_VTK_H

#include "../Support/VtkOutput.h"
#include "ConvexPolyhedron.h"

namespace sdot {

///
struct DisplayVtkConvexPolyhedronParams {
    std::vector<double> cell_values;
    Point<double,3> offset = 0.0;
};

/// display a single cell
template<class Pc,class Opt> void display_vtk( VtkOutput &vo, const ConvexPolyhedron<Pc,2,Opt> &cp, const DisplayVtkConvexPolyhedronParams &params = {} );
template<class Pc,class Opt> void display_vtk( VtkOutput &vo, const ConvexPolyhedron<Pc,3,Opt> &cp, const DisplayVtkConvexPolyhedronParams &params = {} );

} // namespace sdot

#include "display_vtk.tcc"

#endif // DISPLAY_VTK_H
