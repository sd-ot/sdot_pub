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

SimdGraph make_graph( OptParm &opt_parm, const Cp2Lt64CutList &mod ) {
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

    SimdOp *px_0 = gr.make_op( "REG px_0 d 4", {} );
    SimdOp *di_0 = gr.make_op( "REG di_0 d 4", {} );

    SimdOp *px_a = gr.make_op( "AGG", { gr.get_op( px_0, n0 ), gr.get_op( px_0, n2 ) } );
    SimdOp *px_b = gr.make_op( "AGG", { gr.get_op( px_0, n1 ), gr.get_op( px_0, n3 ) } );
    SimdOp *di_a = gr.make_op( "AGG", { gr.get_op( di_0, n0 ), gr.get_op( di_0, n2 ) } );
    SimdOp *di_b = gr.make_op( "AGG", { gr.get_op( di_0, n1 ), gr.get_op( di_0, n3 ) } );

    SimdOp *di_m = gr.make_op( "DIV", { di_a, gr.make_op( "SUB", { di_b, di_a } ) } );
    SimdOp *adds = gr.make_op( "ADD", { px_a, gr.make_op( "MUL", { di_m, gr.make_op( "SUB", { px_a, px_b } ) } ) } );

    std::vector<SimdOp *> r_ch;
    int num_in_adds = switch_cuts;
    for( const Cp2Lt64CutList::Cut &op : mod.ops ) {
        if ( op.single() ) {
            r_ch.push_back( gr.get_op( px_0, op.i0 ) );
        } else {
            r_ch.push_back( gr.get_op( adds, num_in_adds ) );
            num_in_adds ^= 1;
        }
    }

    gr.add_target( gr.make_op( "SET px_0", { gr.make_op( "AGG", r_ch ) } ) );
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
    do {
        Cp2Lt64CutList mod = ref_mod;
        mod.rotate( opt_parm.get_value( mod.ops.size() ) );
        mod.sw( opt_parm.get_value( 1 << mod.split_indices().size() ) );

        graphs.push_back( make_graph( opt_parm, mod ) );
    } while ( opt_parm.inc() );

    // prepare a benchmark
    std::ofstream fout( ".tmp.cpp" );
    fout << "#include \"src/sdot/Support/SimdVec.h\"\n";
    fout << "#include \"src/sdot/Support/Time.h\"\n";
    fout << "#include <iostream>\n";
    fout << "\n";
    fout << "using namespace sdot;\n";
    fout << "using TC = std::uint64_t;\n";
    fout << "using TF = double;\n";
    fout << "using TF4 = SimdVec<TF,4>;\n";
    fout << "using TC4 = SimdVec<TC,4>;\n";
    fout << "\n";
    fout << "void cut_bench( TF *data, const int *cases, int nb_cases ) {\n";
    fout << "    TF4 px_0 = VF::load_aligned( data + 0 * 128 + 0 );\n";
    fout << "    TF4 px_1 = VF::load_aligned( data + 0 * 128 + 4 );\n";
    fout << "    TF4 py_0 = VF::load_aligned( data + 1 * 128 + 0 );\n";
    fout << "    TF4 py_1 = VF::load_aligned( data + 1 * 128 + 4 );\n";
    fout << "    // TC4 pc_0 = VC::load_aligned( nodes.cut_ids + 0 );\n";
    fout << "    // TC4 pc_1 = VC::load_aligned( nodes.cut_ids + 4 );\n";
    fout << "    for( int num_case = 0; ; ++num_case ) {\n";
    fout << "        if ( num_case == nb_cases ) {\n";
    fout << "            VF::store_aligned( data + 0 * 128 + 0, px_0 );\n";
    fout << "            VF::store_aligned( data + 0 * 128 + 4, px_1 );\n";
    fout << "            VF::store_aligned( data + 1 * 128 + 0, py_0 );\n";
    fout << "            VF::store_aligned( data + 1 * 128 + 4, py_1 );\n";
    fout << "            // VC::store_aligned( nodes.cut_ids + 0, pc_0 );\n";
    fout << "            // VC::store_aligned( nodes.cut_ids + 4, pc_1 );\n";
    fout << "            return;\n";
    fout << "        }\n";
    fout << "        \n";
    fout << "        // get distance and outside bit for each node\n";
    fout << "        std::uint16_t nmsk = 1 << nodes_size;\n";
    fout << "        VF cx = cut_dir[ 0 ][ num_cut ];\n";
    fout << "        VF cy = cut_dir[ 1 ][ num_cut ];\n";
    fout << "        VC ci = cut_ids[ num_cut ];\n";
    fout << "        VF cs = cut_ps[ num_cut ];\n";
    fout << "        \n";
    fout << "        VF bi_0 = px_0 * cx + py_0 * cy;\n";
    fout << "        VF bi_1 = px_1 * cx + py_1 * cy;\n";
    fout << "        std::uint16_t outside_nodes = ( ( ( bi_0 > cs ) << 0 ) | ( ( bi_1 > cs ) << 4 ) ) & ( nmsk - 1 );\n";
    fout << "        std::uint16_t case_code = outside_nodes | nmsk;\n";
    fout << "        VF di_0 = bi_0 - cs;\n";
    fout << "        VF di_1 = bi_1 - cs;\n";
    fout << "        \n";
    fout << "        // if nothing has changed => go to the next cut\n";
    fout << "        if ( outside_nodes == 0 )\n";
    fout << "            continue;\n";

    fout << "        \n";
    fout << "        static void *dispatch_table[] = {\n    ";
    for( std::size_t i = 0; i < graphs.size(); ++i )
        fout << "case_" << i << "," << ( i % 8 == 7 || i + 1 == graphs.size() ? "\n    " : " " );
    fout << "    }\n";
    fout << "        goto *dispatch_table[ cases[ num_case ] ];\n";
    for( std::size_t i = 0; i < graphs.size(); ++i ) {
        fout << "      case_" << i << ": {\n";
        graphs[ i ].write_code( fout, "        " );
        fout << "        continue;\n";
        fout << "      }\n";
    }
    fout << "    }\n";
    fout << "}\n";
    fout << "\n";
    fout << "int main() {\n";
    fout << "    std::vector<int> cases( 128, 0 );\n";
    fout << "    alignas (64) TF data[ 3 * 128 ];\n";
    for( std::size_t o = 0; o < 3; ++o )
        for( std::size_t j = 0; j < outside.size(); ++j )
            fout << "    data[ 128 * " << o << " + " << j << " ] = " << ( outside[ j ] ? 1 : -1 ) << ";\n";
    fout << "    cut_bench( TF *data, const int *cases, int nb_cases ) {\n";
    fout << "    std::cout << data[ 0 ] << std::endl;\n";
    fout << "}\n";
    fout << "\n";

    // launch the benchmark
    fout.close();

    // compilation
    cmd( "cd ~/sdot_pub && g++ -O3 -march=native -ffast-math -o .tmp.exe .tmp.cpp" );
    cmd( "./.tmp.exe .tmp.dat" );

    // bench run
    //    cmd( "sudo bash scripts/cpufreq_init.sh" );
    //    cmd( "sudo cset shield --reset" );
    //    cmd( "sudo cset shield --cpu 1-3" ); //  -k on
    //    cmd( "sudo cset shield --shield -v" );
    //    cmd( "sudo cset shield --exec ./.tmp.exe -- .tmp.dat" );
    //    cmd( "sudo cset shield --reset" );

    //    // bench stop
    //    cmd( "sudo bash scripts/cpufreq_stop.sh" );

    return true;
}


int main() {
    make_code( std::cout, { 0, 1, 1, 0 } );
}
