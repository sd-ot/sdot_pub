#include "../Support/TODO.h"
#include "SimdCodegen.h"
#include <fstream>

SimdCodegen::SimdCodegen() {
}

void SimdCodegen::add_possibility( const SimdGraph &gr ) {
    gr_map.push_back( gr );
}

void SimdCodegen::write_code( std::ostream &os ) {
    std::ofstream fout( "/home/leclerc/sdot_pub/.tmp.cpp" );
    fout << "#include \"src/sdot/Support/SimdVec.h\"\n";
    fout << "#include \"src/sdot/Support/Time.h\"\n";
    fout << "#include <iostream>\n";
    fout << "#include <fstream>\n";
    fout << "#include <vector>\n";
    fout << "#include <chrono>\n";
    fout << "\n";
    fout << "//// nsmake cxx_name clang++\n";
    fout << "//// nsmake cpp_flag -march=native\n";
    fout << "//// nsmake cpp_flag -ffast-math\n";
    fout << "//// nsmake cpp_flag -O3\n";
    fout << "\n";
    fout << "using namespace sdot;\n";
    fout << "using CI = std::uint64_t;\n";
    fout << "using TF = double;\n";
    fout << "\n";

    // test
    for( std::size_t cpt = 0; cpt <= gr_map.size(); ++cpt ) {
        fout << "SimdVec<TF,4> __attribute__ ((noinline)) f_" << cpt << "( SimdVec<TF,4> px_0, SimdVec<TF,4> di_0 ) {\n";
        if ( cpt )
            gr_map[ cpt - 1 ].write_code( fout, "        " );
        // fout << "    res = 0;\n";
        fout << "    return px_0;\n";
        fout << "}\n";
    }

    fout << "int main( int /*argc*/, char **argv ) {\n";
    fout << "    std::vector<double> dt;\n";

    std::size_t nb_reps = 50000000ul;
    for( std::size_t cpt = 0; cpt <= gr_map.size(); ++cpt ) {
        fout << "    {\n";
        fout << "        SimdVec<TF,4> px_0{ 1, 2, 3, 4 }, di_0{ 1, 2, 3, 4 };\n";
        fout << "        std::uint64_t t0 = 0, t1 = 0;\n";
        // fout << "     auto t0 = std::chrono::high_resolution_clock::now();\n";
        fout << "        RDTSC_START( t0 );\n";
        fout << "        for( unsigned long i = 0; i < " << nb_reps << "ul; ++i )\n";
        fout << "            px_0 = f_" << cpt << "( px_0, di_0 );\n";
        fout << "        RDTSC_FINAL( t1 );\n";
        // fout << "     auto t1 = std::chrono::high_resolution_clock::now();\n";
        // fout << "     dt.push_back( std::chrono::duration_cast<std::chrono::microseconds>( t1 - t0 ).count() / 1e6 );\n";
        fout << "        dt.push_back( t1 - t0 );\n";
        fout << "        std::cout << \"px: \" << px_0[ 0 ] << \" \" << px_0[ 1 ] << \" \" << px_0[ 2 ] << \" \" << px_0[ 3 ] << std::endl;\n";
        if ( cpt )
            fout << "        std::cout << ( dt.back() - dt[ 0 ] ) / " << nb_reps << " << std::endl;\n";
        else
            fout << "        std::cout << \"ref: \" << dt.back() / " << nb_reps << " << std::endl;\n";
        fout << "    }\n";
    }
    // fout << "    std::ofstream fout( argv[ 1 ] );\n";
    fout << "        std::size_t best = 1, wrst = 1;\n";
    fout << "        for( std::size_t i = 2; i < dt.size(); ++i ) {\n";
    fout << "            if ( dt[ best ] > dt[ i ] ) best = i;\n";
    fout << "            if ( dt[ wrst ] < dt[ i ] ) wrst = i;\n";
    fout << "        }\n";
    fout << "        std::cout << \"best: \" << ( dt[ best ] - dt[ 0 ] ) / " << nb_reps << " << std::endl;\n";
    fout << "        std::cout << \"wrst: \" << ( dt[ wrst ] - dt[ 0 ] ) / " << nb_reps << " << std::endl;\n";
    fout << "        std::cout << \"best num: \" << best << std::endl;\n";
    fout << "}\n";

    //
    fout.close();

    // system( "cd ~/sdot_pub && cat .tmp.cpp" );
    system( "cd ~/sdot_pub && g++ -O3 -march=native -ffast-math -o .tmp.exe .tmp.cpp && ./.tmp.exe .tmp.dat" );
    //    std::ifstream fin( "/home/leclerc/sdot_pub/.tmp.dat" );
    //    double res = 10;
    //    fin >> res;
    //    os << res;
}
