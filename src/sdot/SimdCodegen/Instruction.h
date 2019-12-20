#pragma once

#include "../Support/Display/generic_ostream_output.h"
#include "Reg.h"

///
struct Instruction {
    /**/           Instruction    ();
    virtual       ~Instruction    ();
    virtual void   write_to_stream( std::ostream &os ) = 0;
    virtual double throughput     () const { return 1; }
    virtual double latency        () const { return 1; }
};

///
struct Instruction256Cast128 : public Instruction {
    /**/           Instruction256Cast128( Reg out, Reg inp ) : out( out ), inp( inp ) {}
    virtual void   write_to_stream      ( std::ostream &os ) override { os << "_m128 " << out << " = _mm256_castpd256_pd128( " << inp << " );"; }
    virtual double throughput           () const override { return 0; }
    virtual double latency              () const override { return 0; }
    Reg            out, inp;
};

///
struct Instruction256Extract128 : public Instruction {
    /**/           Instruction256Extract128( Reg out, Reg inp, int extract ) : out( out ), inp( inp ), extract( extract ) {}
    virtual void   write_to_stream         ( std::ostream &os ) override { os << "_m128 " << out << " = _mm256_extractf128_pd( " << inp << ", " << extract << " );"; }
    virtual double latency                 () const override { return 3; }
    Reg            out, inp;
    int            extract;
};

///
struct Instruction256Perm : public Instruction {
    /**/              Instruction256Perm( Reg out, Reg inp, std::array<int,4> nums ) : out( out ), inp( inp ), nums( nums ) {}
    virtual void      write_to_stream   ( std::ostream &os ) override { os << "_m256 " << out << " = _mm256_permute4x64_pd( " << inp << ", " << nums[ 0 ] + 4 * nums[ 1 ] + 16 * nums[ 2 ] + 64 * nums[ 3 ] << " );"; }
    virtual double    latency           () const override { return 3; }
    Reg               out, inp;
    std::array<int,4> nums;
};

///
struct Instruction256DupEven : public Instruction {
    /**/              Instruction256DupEven( Reg out, Reg inp ) : out( out ), inp( inp ) {}
    virtual void      write_to_stream   ( std::ostream &os ) override { os << "_m256 " << out << " = _mm256_movedup_pd( " << inp << " );"; }
    Reg               out, inp;
};

