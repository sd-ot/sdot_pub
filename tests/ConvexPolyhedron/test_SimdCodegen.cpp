#include "../../src/sdot/SimdCodegen/SimdCodegen.h"
#include "../../src/sdot/Support/P.h"
#include <iostream>


int main( int /*argc*/, char **/*argv*/ ) {
    SimdCodegen sc( 4 );

    P( sc.best_path_for( { -1, -1, -1, -1,  0, 1, -1, -1 } ) );
    P( sc.best_path_for( { -1, -1, -1, -1,  2, 3, -1, -1 } ) );
}
