#pragma once

#include "SimdGraph.h"
#include <map>

/**
*/
class SimdCodegen {
public:
    /**/                   SimdCodegen    ();

    void                   add_possibility( const SimdGraph &gr );
    void                   write_code     ( std::ostream &os );

private:
    std::vector<SimdGraph> gr_map;
};


