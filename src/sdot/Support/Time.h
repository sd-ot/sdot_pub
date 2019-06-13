#pragma once

#include <chrono>

/** */
class Time {
public:
    using    TP = std::chrono::high_resolution_clock::time_point;

    double   operator-( const Time &that ) const { return std::chrono::duration_cast<std::chrono::microseconds>( time_point - that.time_point ).count() / 1e6; }

    TP       time_point;
};

inline Time time() {
    return { std::chrono::high_resolution_clock::now() };
}

