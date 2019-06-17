#include "../src/sdot/Geometry/ConvexPolyhedron2.h"
#include "../src/sdot/Support/StaticRange.h"
#include "../src/sdot/Support/ASSERT.h"
#include "../src/sdot/Support/Time.h"
#include "../src/sdot/Support/P.h"
#include "../src/sdot/VtkOutput.h"
#include <map>
using namespace sdot;

// // nsmake cpp_flag -march=skylake
// // nsmake cxx_name clang++

//// nsmake cpp_flag -march=native
//// nsmake lib_flag -O3
//// nsmake cpp_flag -O3

struct Test {
    void write_to_stream( std::ostream &os ) const {
        if ( nb_nodes < 10000 )
            os << "nb_nodes=" << nb_nodes << " nb_cuts=" << nb_cuts << " ";
        if ( use_simd )
            os << ( use_switch ? "simd switch" : "simd tzcnt " );
        else
            os << "generic    ";
    }

    bool operator<( const Test &that ) const {
        return std::tie( nb_nodes, nb_cuts, use_simd, use_switch ) < std::tie( that.nb_nodes, that.nb_cuts, that.use_simd, that.use_switch );
    }

    std::size_t nb_nodes;
    std::size_t nb_cuts;

    bool        use_simd;
    bool        use_switch;
};

template<class Cp,int Simd,int Switch>
void test( std::map<Test,double> &timings, VtkOutput &vo, int &cpt_vo, std::size_t nb_nodes, N<Simd>, N<Switch> ) {
    using Pt = typename Cp::Pt;
    using TF = typename Cp::TF;

    // initial cell
    Cp lc( typename Cp::Box{ { -2, -2 }, { +2, +2 } } );
    constexpr int flags = ConvexPolyhedron::do_not_use_simd     * ( Simd   == 0 ) +
                          ConvexPolyhedron::do_not_use_switches * ( Switch == 0 );
    for( std::size_t i = 0; i < nb_nodes; ++i ) {
        TF a = i * 2 * M_PI / nb_nodes;
        Pt n( cos( a ), sin( a ) );
        lc.plane_cut( n, n, 17, N<flags>() );
    }

    P( nb_nodes, lc );

    lc.display( vo, { TF( cpt_vo ) }, { 2.5 * TF( cpt_vo % 8 ), 2.5 * TF( cpt_vo / 8 ) } );
    ++cpt_vo;;

    //
}

int main() {
    struct Pc {
        enum { store_the_normals = false };
        enum { allow_ball_cut    = false };
        using  TF                = double;
        using  TI                = std::size_t;
        using  CI                = std::size_t;

    };
    using Cp = ConvexPolyhedron2<Pc>;


    int cpt_vo = 0;
    VtkOutput vo( { "smurf" } );
    std::map<Test,double> timings;
    for( std::size_t nb_nodes = 3; nb_nodes < 11; ++nb_nodes )
        test<Cp>( timings, vo, cpt_vo, nb_nodes, /*simd*/ N<0>(), /*switch*/ N<0>() );
    for( std::size_t nb_nodes = 3; nb_nodes < 11; ++nb_nodes )
        test<Cp>( timings, vo, cpt_vo, nb_nodes, /*simd*/ N<1>(), /*switch*/ N<0>() );
    for( std::size_t nb_nodes = 3; nb_nodes < 11; ++nb_nodes )
        test<Cp>( timings, vo, cpt_vo, nb_nodes, /*simd*/ N<1>(), /*switch*/ N<1>() );

    //    constexpr bool want_cmp = true;
    //    for( std::size_t nb_nodes : { /*3, */4 } ) {
    //        for( std::size_t nb_cuts : { 1 /*0, 1, 2, 3*/ } ) {
    //            double rep = nb_cuts == 0 ? 1e7 : 1e6;
    //            StaticRange<1+want_cmp>::for_each( [&]( auto no_simd ) {
    //                StaticRange<1+(no_simd.val==0)*want_cmp>::for_each( [&]( auto no_switch ) {
    //                    constexpr int flags = no_switch.val * ConvexPolyhedron::plane_cut_flag_no_switches +
    //                            no_simd  .val * ConvexPolyhedron::do_not_use_simd;
    //                    Cp lc( Cp::Box{ { 0, 0 }, { 1, 1 } } );
    //                    Cp cp[ 16 ];

    //                    uint64_t t0 = 0, t1 = 0, dt = 0;
    //                    switch ( nb_cuts ) {
    //                    case 0:
    //                        for( std::size_t n = 0; n < rep; ++n ) {
    //                            #pragma unroll
    //                            for( std::size_t i = 0; i < 16; ++i )
    //                                cp[ i ] = lc;
    //                            RDTSC_START( t0 );
    //                            StaticRange<16>::for_each( [&]( auto i ) {
    //                                cp[ i ].plane_cut( { 2.0, 0.5 }, { 1.0, 0.0 }, 17, N<flags>() );
    //                            } );
    //                            RDTSC_FINAL( t1 );
    //                            dt += t1 - t0;
    //                        }
    //                        break;
    //                    case 1:
    //                        for( std::size_t n = 0; n < rep; ++n ) {
    //                            #pragma unroll
    //                            for( std::size_t i = 0; i < 16; ++i )
    //                                cp[ i ] = lc;
    //                            RDTSC_START( t0 );
    //                            StaticRange<16>::for_each( [&]( auto i ) {
    //                                cp[ i ].plane_cut( { 0.25, 0.75 }, { -1.0, 1.0 }, 17, N<flags>() );
    //                            } );
    //                            RDTSC_FINAL( t1 );
    //                            dt += t1 - t0;
    //                        }
    //                        break;
    //                    case 2:
    //                        for( std::size_t n = 0; n < rep; ++n ) {
    //                            #pragma unroll
    //                            for( std::size_t i = 0; i < 16; ++i )
    //                                cp[ i ] = lc;
    //                            RDTSC_START( t0 );
    //                            StaticRange<16>::for_each( [&]( auto i ) {
    //                                cp[ i ].plane_cut( { 0.5, 0.5 }, { 1.0, 0.0 }, 17, N<flags>() );
    //                            } );
    //                            RDTSC_FINAL( t1 );
    //                            dt += t1 - t0;
    //                        }
    //                        break;
    //                    case 3:
    //                        for( std::size_t n = 0; n < rep; ++n ) {
    //                            #pragma unroll
    //                            for( std::size_t i = 0; i < 16; ++i )
    //                                cp[ i ] = lc;
    //                            RDTSC_START( t0 );
    //                            StaticRange<16>::for_each( [&]( auto i ) {
    //                                cp[ i ].plane_cut( { 0.25, 0.25 }, { 1.0, 1.0 }, 17, N<flags>() );
    //                            } );
    //                            RDTSC_FINAL( t1 );
    //                            dt += t1 - t0;
    //                        }
    //                        break;
    //                    default:
    //                        TODO;
    //                    }

    //                    Test t;
    //                    t.use_switch = ! no_switch.val;
    //                    t.use_simd   = ! no_simd.val;
    //                    t.nb_nodes   = nb_nodes;
    //                    t.nb_cuts    = nb_cuts;

    //                    timings[ t ] = dt / 16.0 / rep;

    //                    // display
    //                    static double inc = 0;
    //                    cp[ 0 ].display( vo, { 1.0 }, { inc, 0.0 } );
    //                    P( cp[ 0 ] );
    //                    inc += 1.5;
    //                } );
    //            } );
    //        }
    //    }

    // raw timings
    for( auto v : timings )
        std::cout << v.first << " -> " << v.second << " cycles/cell\n";

    std::map<std::size_t,double> freq_nb_cut;
    freq_nb_cut[ 0 ] = 0.646137;
    freq_nb_cut[ 1 ] = 0.145905;
    freq_nb_cut[ 2 ] = 0.139458;
    freq_nb_cut[ 3 ] = 0.054996;
    freq_nb_cut[ 4 ] = 0.011352;
    freq_nb_cut[ 5 ] = 0.001821;
    freq_nb_cut[ 6 ] = 0.000274;

    //
    std::map<Test,double> synth_timings;
    for( auto v : timings ) {
        Test t = v.first;
        t.nb_nodes = 10000;
        t.nb_cuts = 10000;
        synth_timings[ t ] += v.second * freq_nb_cut[ v.first.nb_cuts ];
    }

    std::cout << "-------------------------------\n";
    for( auto v : synth_timings )
        std::cout << v.first << " -> " << v.second << " cycles/cell\n";

    vo.save( "vtk/pd.vtk" );
}
