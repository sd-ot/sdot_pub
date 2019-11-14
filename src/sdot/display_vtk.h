#pragma once

#include "Integration/SpaceFunctions/Constant.h"
#include "Integration/FunctionEnum.h"
#include "Support/VtkOutput.h"
#include <mutex>

namespace sdot {

/**
   We assume that grid has already been initialized by diracs
*/
template<class Domain,class Grid,class Func>
void display_vtk( Domain &domain, Grid &grid, VtkOutput &res, const Func &/*radial_func*/ ) {
    std::mutex m;
    domain.for_each_laguerre_cell( grid, [&]( auto &lc, int /*num_thread*/, auto /*space_func*/ ) {
        m.lock();
        lc.display( res, { VtkOutput::TF( *lc.dirac_index ) } );
        m.unlock();
    } ); // , radial_func.need_ball_cut()
}

template<class Domain,class Grid>
void display_vtk( Domain &domain, Grid &grid, VtkOutput &vo ) {
    display_vtk( domain, grid, vo, FunctionEnum::Unit() );
}


template<class Grid,class Domain>
void display_vtk( Domain &domain, Grid &grid, std::string res ) {
    VtkOutput vo;
    display_vtk( domain, grid, vo );
    vo.save( res );
}

} // namespace sdot
