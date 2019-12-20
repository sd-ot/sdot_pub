#pragma once

#include <vector>
#include <string>

/**
*/
class SimdOp {
public:
    /* */                 SimdOp         ( std::string name, const std::vector<SimdOp *> &children );

    void                  write_to_stream( std::ostream &os ) const;

    std::vector<SimdOp *> children;
    std::uint64_t         op_id;
    std::string           name;
};

