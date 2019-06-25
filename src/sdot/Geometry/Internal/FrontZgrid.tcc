#include "../../Support/ASSERT.h"
#include "FrontZgrid.h"
#include <cmath>

namespace sdot {

template<class ZG>
FrontZgrid<ZG>::FrontZgrid( TI &op_count, std::vector<TI> &visited ) : op_count( op_count ), visited( visited ) {
}

template<class ZG>
void FrontZgrid<ZG>::init( TI num_cell, Pt position, TF weight ) {
    ++op_count;
    set_visited( num_cell );

    orig_position = position;
    orig_weight = weight;
}

template<class ZG>
void FrontZgrid<ZG>::set_visited( TI num_cell ) {
    visited[ num_cell ] = op_count;
}

template<class ZG> template<class Cell>
typename FrontZgrid<ZG>::TF FrontZgrid<ZG>::dist( const Cell &cell ) {
    return norm_2_p2( cell.pos - orig_position );
}

template<class ZG> template<class Grid>
void FrontZgrid<ZG>::push_without_check( TI num_cell, const Grid &grid ) {
    items.push( Item{ num_cell, dist( grid.cells[ num_cell ] ) } );
    set_visited( num_cell );
}

template<class ZG> template<class Grid>
void FrontZgrid<ZG>::push( TI num_cell, const Grid &grid ) {
    if ( visited[ num_cell ] != op_count )
        push_without_check( num_cell, grid );
}

template<class ZG>
typename FrontZgrid<ZG>::Item FrontZgrid<ZG>::pop() {
    Item res = items.top();
    items.pop();
    return res;
}

template<class ZG>
bool FrontZgrid<ZG>::empty() const {
    return items.empty();
}

}
