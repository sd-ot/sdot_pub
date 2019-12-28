#pragma once

#include <vector>

/**
*/
class OptParm {
public:
    struct             Value       { std::size_t val, max; };

    /**/               OptParm     ();

    double             completion  () const;
    std::size_t        get_value   ( std::size_t max );
    void               restart     ();
    bool               inc         (); ///< return false if finished

    std::vector<Value> previous_values;
    std::vector<Value> current_values;
    std::size_t        random;
    std::size_t        count;
};

