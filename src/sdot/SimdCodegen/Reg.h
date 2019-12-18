#pragma once

#include <ostream>
#include <array>

class Reg {
public:
    /**/        Reg            ( int size, std::string id );

    void        write_to_stream( std::ostream &os ) const;
    bool        operator==     ( const Reg &that ) const;

    int         size;
    std::string id;
};

