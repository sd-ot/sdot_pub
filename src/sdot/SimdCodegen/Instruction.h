#pragma once

#include <ostream>
#include <array>

///
struct Instruction {
    /**/           Instruction    ();
    virtual       ~Instruction    ();
    virtual void   write_to_stream( std::ostream &os, std::string prefix = "di" ) = 0;
    virtual double throughput     () const { return 1; }
    virtual double latency        () const { return 1; }
};

///
struct Instruction256Cast128 : public Instruction {
    /**/           Instruction256Cast128( int out, int inp ) : out( out ), inp( inp ) {}
    virtual void   write_to_stream      ( std::ostream &os, std::string prefix ) override { os << "_m128 " << prefix << "_" << out << " = _mm256_castpd256_pd128( " << prefix << "_" << inp << " )"; }
    virtual double throughput           () const override { return 0; }
    virtual double latency              () const override { return 0; }
    int            out, inp;
};

///
struct Instruction256Extract128 : public Instruction {
    /**/           Instruction256Extract128( int out, int inp, int extract ) : out( out ), inp( inp ), extract( extract ) {}
    virtual void   write_to_stream         ( std::ostream &os, std::string prefix ) override { os << "_m128 " << prefix << "_" << out << " = _mm256_extractf128_pd( " << prefix << "_" << inp << ", " << extract << " )"; }
    virtual double latency                 () const override { return 3; }
    int            out, inp;
    int            extract;
};

///
struct Instruction256Perm : public Instruction {
    /**/              Instruction256Perm( int out, int inp, std::array<int,4> nums ) : out( out ), inp( inp ), nums( nums ) {}
    virtual void      write_to_stream   ( std::ostream &os, std::string prefix ) override { os << "_m256 " << prefix << "_" << out << " = _mm256_permute4x64_pd( " << prefix << "_" << inp << ", " << nums[ 0 ] + 4 * nums[ 1 ] + 16 * nums[ 2 ] + 64 * nums[ 3 ] << " )"; }
    virtual double    latency           () const override { return 3; }
    int               out, inp;
    std::array<int,4> nums;
};

