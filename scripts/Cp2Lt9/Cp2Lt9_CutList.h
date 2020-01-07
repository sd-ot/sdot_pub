#pragma once

#include "../../src/sdot/Support/P.h"
#include <vector>

/**
  Used for code generation
*/
struct Cp2Lt9_CutList {
    struct                   Cut {
        /**/                 Cut            ( std::size_t i0 = 0, std::size_t i1 = 0, int dir = 0 ) : dir( dir ), i0( i0 ), i1( i1 ), sw( false ) {}

        void                 write_to_stream( std::ostream &os ) const { if ( single() ) os << inside_node(); else os << "[" << i0 << "," << i1 << "]"; }
        bool                 going_outside  () const { return dir  < 0; }
        bool                 going_inside   () const { return dir  > 0; }
        bool                 single         () const { return dir == 0; }
        bool                 split          () const { return dir != 0; }

        std::size_t          outside_node   () const { return dir > 0 ? i0 : i1; }
        std::size_t          inside_node    () const { return dir > 0 ? i1 : i0; }

        std::size_t          n0             () const { return sw ? i1 : i0; }
        std::size_t          n1             () const { return sw ? i0 : i1; }

        int                  dir;           ///< -1 => going outside. 0 => single node. +1 => going inside.
        std::size_t          i0;            ///<
        std::size_t          i1;            ///<
        bool                 sw;            ///<
    };

    /**/                     Cp2Lt9_CutList ( std::vector<bool> outside = {} );

    void                     write_to_stream( std::ostream &os ) const;
    std::vector<std::size_t> split_indices  () const;
    void                     rotate         ( std::size_t off );
    void                     sw             ( std::uint64_t val );

    std::vector<Cut>         ops;
};
