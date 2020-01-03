#include "../src/sdot/SimdCodegen/SimdGraph.h"
#include "../src/sdot/Support/OptParm.h"
#include "../src/sdot/Support/ASSERT.h"
#include "../src/sdot/Support/ERROR.h"
#include "Cp2Lt64CutList.h"
#include <fstream>

void cmd( std::string c ) {
    std::cout << c << std::endl;
    if ( system( c.c_str() ) )
        ERROR( "..." );
}

SimdGraph make_graph( OptParm &opt_parm, const Cp2Lt64CutList &mod, int simd_size ) {
    std::vector<std::size_t> sp_ind = mod.split_indices();
    ASSERT( sp_ind.size() == 2, "" );
    SimdGraph gr;

    int n0 = mod.ops[ sp_ind[ 0 ] ].n0(), n1 = mod.ops[ sp_ind[ 0 ] ].n1();
    int n2 = mod.ops[ sp_ind[ 1 ] ].n0(), n3 = mod.ops[ sp_ind[ 1 ] ].n1();
    bool switch_cuts = opt_parm.get_value( 2 );
    if ( switch_cuts ) {
        std::swap( n0, n2 );
        std::swap( n1, n3 );
    }

    std::vector<SimdOp *> px_s, py_s, di_s;
    for( std::size_t i = 0; i < mod.ops.size(); i += simd_size ) {
        px_s.push_back( gr.make_op( "REG px_" + std::to_string( i / simd_size ) + " d " + std::to_string( simd_size ), {} ) );
        py_s.push_back( gr.make_op( "REG py_" + std::to_string( i / simd_size ) + " d " + std::to_string( simd_size ), {} ) );
        di_s.push_back( gr.make_op( "REG di_" + std::to_string( i / simd_size ) + " d " + std::to_string( simd_size ), {} ) );
    }

    SimdOp *px_a = gr.make_op( "AGG", { gr.get_op( px_s[ n0 / simd_size ], n0 % simd_size ), gr.get_op( px_s[ n2 / simd_size ], n2 % simd_size ) } );
    SimdOp *px_b = gr.make_op( "AGG", { gr.get_op( px_s[ n1 / simd_size ], n1 % simd_size ), gr.get_op( px_s[ n3 / simd_size ], n3 % simd_size ) } );
    SimdOp *di_a = gr.make_op( "AGG", { gr.get_op( di_s[ n0 / simd_size ], n0 % simd_size ), gr.get_op( di_s[ n2 / simd_size ], n2 % simd_size ) } );
    SimdOp *di_b = gr.make_op( "AGG", { gr.get_op( di_s[ n1 / simd_size ], n1 % simd_size ), gr.get_op( di_s[ n3 / simd_size ], n3 % simd_size ) } );

    SimdOp *di_m = gr.make_op( "DIV", { di_a, gr.make_op( "SUB", { di_b, di_a } ) } );
    SimdOp *adds = gr.make_op( "ADD", { px_a, gr.make_op( "MUL", { di_m, gr.make_op( "SUB", { px_a, px_b } ) } ) } );

    std::vector<std::vector<SimdOp *>> r_ch( ( mod.ops.size() + simd_size - 1 ) / simd_size );
    int num_in_adds = switch_cuts;
    for( std::size_t i = 0; i < mod.ops.size(); ++i ) {
        if ( mod.ops[ i ].single() ) {
            r_ch[ i / simd_size ].push_back( gr.get_op( px_s[ mod.ops[ i ].i0 / simd_size ], mod.ops[ i ].i0 % simd_size ) );
        } else {
            r_ch[ i / simd_size ].push_back( gr.get_op( adds, num_in_adds ) );
            num_in_adds ^= 1;
        }
    }

    for( std::size_t i = 0; i < r_ch.size(); ++i )
        gr.add_target( gr.make_op( "SET px_" + std::to_string( i ), { gr.make_op( "AGG", r_ch[ i ] ) } ) );

    gr.set_msg( va_string( "mod={}, swith_cuts={}", mod, switch_cuts ) );
    //    gr.write_code( std::cout, "    " );
    //    gr.display();

    return gr;
}

bool make_code( std::ostream &os, std::vector<bool> outside ) {
    // make a ref Mod
    Cp2Lt64CutList ref_mod;
    for( std::size_t i = 0; i < outside.size(); ++i ) {
        if ( outside[ i ] )
            continue;

        // going inside
        std::size_t h = ( i + outside.size() - 1 ) % outside.size();
        if ( outside[ h ] )
            ref_mod.ops.push_back( { h, i, 1 } );

        // inside point
        ref_mod.ops.push_back( { i, i, 0 } );

        // outside point => create points on boundaries
        std::size_t j = ( i + 1 ) % outside.size();
        if ( outside[ j ] )
            ref_mod.ops.push_back( { i, j, 1 } );
    }

    // everything is outside
    if ( ref_mod.ops.empty() ) {
        os << "        // everything is outside\n";
        os << "        nodes_size = 0;\n";
        os << "        return fu( *this );\n";
        return true;
    }

    // everything is inside
    if ( ref_mod.split_indices().empty() ) {
        os << "        // everything is inside\n";
        os << "        continue;\n";
        return true;
    }

    // uncommon cases (to reduce code size)
    if ( ref_mod.split_indices().size() > 2 ) {
        return false;
    }

    // => make a list of graphs
    std::vector<SimdGraph> graphs;
    OptParm opt_parm;
    int simd_size = 2;
    do {
        Cp2Lt64CutList mod = ref_mod;
        mod.rotate( opt_parm.get_value( mod.ops.size() ) );
        mod.sw( opt_parm.get_value( 1 << mod.split_indices().size() ) );

        graphs.push_back( make_graph( opt_parm, mod, simd_size ) );
    } while ( opt_parm.inc() );

    // prepare a benchmark
    std::ofstream fout( ".tmp.cpp" );
    fout << "#include \"scripts/bench_Cp2Lt64_code.h\"\n";
    fout << "\n";
    fout << "using namespace sdot;\n";
    fout << "using TC = std::uint64_t;\n";
    fout << "using TF = double;\n";
    fout << "\n";
    fout << "double __attribute__ ((noinline)) cut_bench( int *cases_data, int cases_size ) {\n";
    for( std::size_t i = 0; i < outside.size(); i += simd_size )
        fout << "    SimdVec<TF," << simd_size << "> px_" << i / simd_size << " = 0, py_" << i / simd_size << " = 0, di_" << i / simd_size << ";\n";
    for( std::size_t j = 0; j < outside.size(); ++j )
        fout << "    di_" << j / simd_size << "[ " << j % simd_size << " ] = " << ( outside[ j ] ? 1 : -1 ) << ";\n";
    fout << "    for( int num_case = 0; ; ++num_case ) {\n";
    fout << "        if ( num_case == cases_size ) {\n";
    fout << "            return px_0[ 0 ] + py_0[ 0 ];\n";
    fout << "        }\n";
    fout << "        \n";
    fout << "        static void *dispatch_table[] = {\n            ";
    for( std::size_t i = 0; i < 2 + graphs.size(); ++i )
        fout << "&&case_" << i << "," << ( i % 8 == 7 || i + 1 == graphs.size() ? "\n            " : " " );
    fout << "};\n";
    fout << "        goto *dispatch_table[ cases_data[ num_case ] ];\n";
    fout << "      case_0: {\n";
    for( std::size_t j = 0; j < outside.size(); ++j ) {
        fout << "        px_" << j / 4 << "[ " << j % 4 << " ] = " << ( outside[ j ] ? 1 : -1 ) << ";\n";
        fout << "        py_" << j / 4 << "[ " << j % 4 << " ] = " << ( outside[ j ] ? 1 : -1 ) << ";\n";
    }
    fout << "        continue;\n";
    fout << "      }\n";
    fout << "      case_1: {\n";
    for( std::size_t j = 0; j < outside.size(); ++j ) {
        fout << "        px_" << j / 4 << "[ " << j % 4 << " ] = " << ( outside[ j ] ? 1 : -1 ) << ";\n";
        fout << "        py_" << j / 4 << "[ " << j % 4 << " ] = " << ( outside[ j ] ? 1 : -1 ) << ";\n";
    }
    fout << "        continue;\n";
    fout << "      }\n";
    for( std::size_t i = 0; i < graphs.size(); ++i ) {
        fout << "      case_" << i + 2 << ": {\n";
        graphs[ i ].write_code( fout, "        " );
        fout << "        continue;\n";
        fout << "      }\n";
    }
    fout << "    }\n";
    fout << "}\n";
    fout << "\n";
    fout << "int main( int argc, char **argv ) {\n";
    fout << "    bench_Cp2Lt64_code( cut_bench, " << graphs.size() << ", argc >= 2 ? argv[ 1 ] : nullptr );\n";
    fout << "}\n";
    fout << "\n";

    // launch the benchmark
    fout.close();

    cmd( "cd ~/sdot_pub && g++ -O3 -march=native -ffast-math -o .tmp.exe .tmp.cpp" );
    cmd( "./.tmp.exe .tmp.dat" );

    // get best graph
    std::ifstream fin( ".tmp.dat" );
    std::size_t best_gr;
    fin >> best_gr;

    graphs[ best_gr ].write_code( os, "        " );

    return true;
}


int main() {
    make_code( std::cout, { 0, 1, 1, 0 } );
}
