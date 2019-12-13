#include "../../src/sdot/ConvexPolyhedron/ConvexPolyhedron.h"
#include "../../src/sdot/Support/VtkOutput.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/P.h"
using namespace sdot;

template<class Cp>
void test_3d_lt64() {
    using Pt = typename Cp::Pt;

    Cp cp( Pt( 0.0 ), Pt( 1.0 ) );
    P( cp );
}

int main() {
    struct Pc {
        enum { dim = 4 };
        using  CI  = int;
        using  TF  = double;
    };

    test_3d_lt64<ConvexPolyhedron<Pc>>();
}
