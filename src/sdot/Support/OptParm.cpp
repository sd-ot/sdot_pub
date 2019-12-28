#include "OptParm.h"
#include "random"

OptParm::OptParm() {
    random = 0;
    count = 0;
}

double OptParm::completion() const {
    if ( random )
        return 1.0 * count / random;
    double res = 0, mul = 1;
    for( const Value &v : current_values ) {
        mul /= v.max;
        res += mul * v.val;
    }
    return res;
}

std::size_t OptParm::get_value( std::size_t max ) {
    if ( max <= 1 )
        return 0;

    if ( random )
        return std::size_t( rand() ) % max;

    if ( current_values.size() < previous_values.size() )
        current_values.push_back( previous_values[ current_values.size() ] );
    else
        current_values.push_back( { 0ul, max } );

    return current_values.back().val;
}

void OptParm::restart() {
    previous_values = std::move( current_values );
}

bool OptParm::inc() {
    restart();
    if ( random )
        return ++count < random;

    while ( previous_values.size() ) {
        auto &p = previous_values.back();
        if ( ++p.val < p.max )
            return true;
        previous_values.pop_back();
    }
    return false;
}
