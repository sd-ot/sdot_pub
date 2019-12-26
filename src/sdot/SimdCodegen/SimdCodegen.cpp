#include "../Support/TODO.h"
#include "SimdCodegen.h"
#include <fstream>

SimdCodegen::SimdCodegen() {
}

void SimdCodegen::add_possibility( const SimdGraph &gr ) {
    gr_map[ "" ] = gr;
}

void SimdCodegen::write_code( std::ostream &os ) {
    std::ofstream fout( "/home/leclerc/sdot_pub/.tmp.cpp" );
    fout << "#include \"src/sdot/Support/SimdVec.h\"\n";
    fout << "#include <iostream>\n";
    fout << "#include <fstream>\n";
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
    int cpt = 0;
    for( auto &p : gr_map ) {
        fout << "SimdVec<TF,4> __attribute__ ((noinline)) f_" << cpt++ << "( SimdVec<TF,4> px_0, SimdVec<TF,4> di_0, int &res ) {\n";
        fout << "    for( unsigned long i = 0; i < 100000000ul; ++i ) {\n";
        p.second.write_code( fout, "        " );
        fout << "    }\n";
        fout << "    res = 0;\n";
        fout << "    return px_0;\n";
        fout << "}\n";
    }

    fout << "int main( int /*argc*/, char **argv ) {\n";
    fout << "    int res = 0;\n";
    cpt = 0;
    for( auto &p : gr_map ) {
        fout << "    auto t0 = std::chrono::high_resolution_clock::now();\n";
        fout << "    f_" << cpt++ << "( TF( 0 ), { 1, 2, 3, 4 }, res );\n";
        fout << "    auto t1 = std::chrono::high_resolution_clock::now();\n";
    }
    fout << "    std::ofstream fout( argv[ 1 ] );\n";
    fout << "    fout << std::chrono::duration_cast<std::chrono::microseconds>( t1 - t0 ).count() / 1e6 << std::endl;\n";
    fout << "    return res;\n";
    fout << "}\n";

    //
    fout.close();

    system( "cd ~/sdot_pub && cat .tmp.cpp" );
    system( "cd ~/sdot_pub && clang++ -O3 -march=native -ffast-math -o .tmp.exe .tmp.cpp && ./.tmp.exe .tmp.dat" );
    std::ifstream fin( "/home/leclerc/sdot_pub/.tmp.dat" );
    double res = 10;
    fin >> res;
    os << res;
}
