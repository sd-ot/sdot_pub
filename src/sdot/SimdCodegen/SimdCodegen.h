#pragma once

#include "SimdGraph.h"
#include <map>

/**
*/
class SimdCodegen {
public:
    /**/                            SimdCodegen    ();

    void                            add_possibility( const SimdGraph &gr );
    void                            write_code     ( std::ostream &os );

private:
    std::map<std::string,SimdGraph> gr_map;
};


