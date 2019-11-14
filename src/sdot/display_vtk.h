#pragma once

#include "Integration/SpaceFunctions/Constant.h"
#include "Integration/FunctionEnum.h"
#include "Support/VtkOutput.h"
#include <mutex>

namespace sdot {

/**
   We assume that grid has already been initialized by diracs
*/
template<class Grid,class Domain,class Func>
void display_vtk( VtkOutput &res, Grid &grid, Domain &domain, const Func &/*radial_func*/ ) {
    std::mutex m;
    grid.for_each_laguerre_cell( [&]( auto &lc, int /*num_thread*/ ) {
        domain.for_each_intersection( lc, [&]( auto &cp, SpaceFunctions::Constant<TF> /*space_func*/ ) {
            m.lock();
            cp.display( res, { TF( *cp.dirac_index ) } );
            m.unlock();
        } );
    }, domain.englobing_convex_polyhedron(), N<0>(), false ); // , radial_func.need_ball_cut()
}

template<class Grid,class Domain>
void display_vtk( VtkOutput &res, Grid &grid, Domain &domain ) {
    display_vtk( res, grid, domain, FunctionEnum::Unit() );
}


template<class Grid,class Domain>
void display_vtk( std::string res, Grid &grid, Domain &domain ) {
    VtkOutput vo;
    display_vtk( vo, grid, domain );
    vo.save( res );
}

} // namespace sdot
