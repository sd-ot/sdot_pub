#include "../../src/sdot/SimdCodegen/SimdCodegen.h"
#include "../../src/sdot/Support/P.h"
#include <iostream>


int main( int /*argc*/, char **/*argv*/ ) {
    SimdCodegen sc( 4 );

    Reg i4( 4, "i" ), o2( 2, "o" );
    PN( sc.best_path_for( o2, { { i4, 0 }, { i4, 1 } } ) );
    PN( sc.best_path_for( o2, { { i4, 2 }, { i4, 3 } } ) );
    PN( sc.best_path_for( o2, { { i4, 0 }, { i4, 2 } } ) );
}
