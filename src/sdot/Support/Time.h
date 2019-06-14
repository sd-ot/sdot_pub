#pragma once

#include <chrono>

namespace sdot {

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

#define RDTSC_START(cycles)                                                   \
    do {                                                                      \
        unsigned cyc_high, cyc_low;                                           \
        __asm volatile(                                                       \
            "cpuid\n\t"                                                       \
            "rdtsc\n\t"                                                       \
            "mov %%edx, %0\n\t"                                               \
            "mov %%eax, %1\n\t"                                               \
            : "=r"(cyc_high), "=r"(cyc_low)::"%rax", "%rbx", "%rcx", "%rdx"); \
        (cycles) = ((uint64_t)cyc_high << 32) | cyc_low;                      \
        __asm volatile("" ::: /* pretend to clobber */ "memory");             \
    } while (0)

#define RDTSC_FINAL(cycles)                                                   \
    do {                                                                      \
        unsigned cyc_high, cyc_low;                                  \
        __asm volatile(                                                       \
            "rdtscp\n\t"                                                      \
            "mov %%edx, %0\n\t"                                               \
            "mov %%eax, %1\n\t"                                               \
            "cpuid\n\t"                                                       \
            : "=r"(cyc_high), "=r"(cyc_low)::"%rax", "%rbx", "%rcx", "%rdx"); \
        (cycles) = ((uint64_t)cyc_high << 32) | cyc_low;                      \
    } while (0)

}
