#include "../src/sdot/Geometry/ConvexPolyhedron2.h"
#include "../src/sdot/Support/StaticRange.h"
#include "../src/sdot/Support/ASSERT.h"
#include "../src/sdot/Support/Time.h"
#include "../src/sdot/Support/P.h"
#include "../src/sdot/VtkOutput.h"
#include <map>
using namespace sdot;

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
            os << " generic   ";
    }

    bool operator<( const Test &that ) const {
        return std::tie( nb_nodes, nb_cuts, use_simd, use_switch ) < std::tie( that.nb_nodes, that.nb_cuts, that.use_simd, that.use_switch );
    }

    std::size_t nb_nodes;
    std::size_t nb_cuts;

    bool        use_simd;
    bool        use_switch;
};

int main() {
    struct Pc {
        using TF = double;
        using TI = std::size_t;
        using CI = std::size_t;

    };
    using Cp = ConvexPolyhedron2<Pc>;

    VtkOutput vo( { "smurf" } );
    std::map<Test,double> timings;
    for( std::size_t nb_nodes : { /*3, */4 } ) {
        for( std::size_t nb_cuts : { 0, 2 } ) {
            double rep = nb_cuts == 0 ? 1e8 : 1e7;
            StaticRange<2>::for_each( [&]( auto no_simd ) {
                StaticRange<1+(no_simd.val==0)>::for_each( [&]( auto no_switch ) {
                    constexpr int flags = no_switch.val * ConvexPolyhedron::plane_cut_flag_no_switches +
                                          no_simd  .val * ConvexPolyhedron::do_not_use_simd;
                    Cp cp( Cp::Box{ { 0, 0 }, { 1, 1 } } );

                    Time t0 = time();
                    switch ( nb_cuts ) {
                    case 0:
                        for( double x = 0.5; x > 0; x -= 0.5 / rep )
                            cp.plane_cut( { 2 + x, 0.5 }, { 1.0, 0.0 }, 17, N<flags>() );
                        break;
                    case 2:
                        for( double x = 0.5; x > 0; x -= 0.5 / rep )
                            cp.plane_cut( { x, 0.5 }, { 1.0, 0.0 }, 17, N<flags>() );
                        break;
                    default:
                        TODO;
                    }
                    double dt = ( time() - t0 ) / rep;

                    Test t;
                    t.use_switch = ! no_switch.val;
                    t.use_simd   = ! no_simd.val;
                    t.nb_nodes   = nb_nodes;
                    t.nb_cuts    = nb_cuts;
                    timings[ t ] = dt;

                    if ( rep == 1 ) {
                        cp.display( vo, { 1.0 } );
                        P( cp );
                    }
                } );
            } );
        }
    }

    // raw timings
    for( auto v : timings )
        P( v.first, v.second );

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

    P( "-------------" );
    for( auto v : synth_timings )
        P( v.first, v.second );

    vo.save( "vtk/pd.vtk" );
}
