#include "../src/sdot/SimdCodegen/SimdGraph.h"
#include "../src/sdot/Support/OptParm.h"
#include "../src/sdot/Support/ASSERT.h"
#include "../src/sdot/Support/ERROR.h"
#include "Cp2Lt64CutList.h"
#include <fstream>
#include <map>

void cmd( std::string c ) {
    std::cout << c << std::endl;
    if ( system( c.c_str() ) )
        ERROR( "..." );
}

SimdGraph make_graph( OptParm &opt_parm, const Cp2Lt64CutList &mod, std::vector<bool> outside, int simd_size ) {
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
    for( std::size_t i = 0; i < outside.size(); i += simd_size ) {
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

    for( std::size_t i = 0; i < r_ch.size(); ++i ) {
        while ( r_ch[ i ].size() < simd_size )
            r_ch[ i ].push_back( r_ch[ i ].back() );
        gr.add_target( gr.make_op( "SET px_" + std::to_string( i ), { gr.make_op( "AGG", r_ch[ i ] ) } ) );
    }

    gr.set_msg( va_string( "mod={}, swith_cuts={}", mod, switch_cuts ) );
    //    gr.write_code( std::cout, "    " );
    //    gr.display();

    return gr;
}

bool make_code( SimdGraph &res, unsigned case_code, int simd_size, int max_nb_nodes ) {
    std::vector<bool> outside;
    for( unsigned cp = case_code; cp; cp /= 2 )
        outside.push_back( cp & 1 );
    outside.pop_back();
    P( simd_size, outside );

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
        res.msg  = "everything is outside";
        res.prel = "nodes_size = 0;";
        res.suff = "return fu( *this );";
        return true;
    }

    // everything is inside
    if ( ref_mod.split_indices().empty() ) {
        res.msg = "everything is inside";
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

        graphs.push_back( make_graph( opt_parm, mod, outside, simd_size ) );
    } while ( opt_parm.inc() );

    // prepare a benchmark
    std::ofstream fout( ".tmp.cpp" );
    fout << "#include \"scripts/bench_Cp2Lt64_code.h\"\n";
    fout << "\n";
    fout << "using namespace sdot;\n";
    fout << "using TF = double;\n";
    fout << "using TC = std::uint64_t;\n";
    fout << "using VF = SimdVec<TF," << simd_size << ">;\n";
    fout << "\n";
    for( std::size_t num_graph = 0; num_graph < graphs.size(); ++num_graph ) {
        // function signature
        fout << "void cut_bench_" << num_graph << "( TF *px, TF *py, int &nodes_size, const TF *cut_x, const TF *cut_y, const TF *cut_s, int cn ) {\n";

        // load px, py, pi, ...
        for( char c : std::string( "xy" ) )
            for( std::size_t i = 0; i < max_nb_nodes + 1 /*we may have to create a new point*/; i += simd_size )
                fout << "    SimdVec<TF," << simd_size << "> p" << c << "_" << i / simd_size << " = SimdVec<TF," << simd_size << ">::load_aligned( p" << c << " + " << i << " );\n";

        // loop over the cuts
        fout << "    for( int num_cut = 0; num_cut < cn; ++num_cut ) {\n";

        // get distance and outside bit for each node
        fout << "        int nmsk = 1 << nodes_size;\n";
        fout << "        // VC ci = cut_i[ num_cut ];\n";
        fout << "        VF cx = cut_x[ num_cut ];\n";
        fout << "        VF cy = cut_y[ num_cut ];\n";
        fout << "        VF cs = cut_s[ num_cut ];\n";
        fout << "        \n";
        for( std::size_t i = 0; i < max_nb_nodes; i += simd_size )
            fout << "        VF bi_" << i / simd_size << " = px_" << i / simd_size << " * cx + py_" << i / simd_size << " * cy;\n";
        fout << "        int outside_nodes = ";
        for( std::size_t i = 0; i < max_nb_nodes; i += simd_size )
            fout << ( i ? " | " : "" ) << "( ( bi_" << i / simd_size << " > cs ) << " << i << " )";
        fout << ";\n";
        fout << "        int case_code = ( outside_nodes & ( nmsk - 1 ) ) | nmsk;\n";
        for( std::size_t i = 0; i < max_nb_nodes; i += simd_size )
            fout << "        VF di_" << i / simd_size << " = bi_" << i / simd_size << " - cs;\n"; // TODO: test if di > 0
        fout << "        \n";
        fout << "        // if nothing has changed => go to the next cut\n";
        fout << "        if ( outside_nodes == 0 )\n";
        fout << "            continue;\n";

        // dispatch
        fout << "        static void *dispatch_table[] = { ";
        for( int i = 0; i < case_code; ++i )
            fout << "&&case_init, ";
        fout << "&&case_cut };\n";
        fout << "        goto *dispatch_table[ case_code ];\n";

        // init case
        fout << "        case_init: {\n";
        for( std::size_t j = 0; j < outside.size(); ++j )
            fout << "            px_" << j / simd_size << "[ " << j % simd_size << " ] = " << ( outside[ j ] ? 1 : -1 ) << ";\n";
        fout << "            continue;\n";
        fout << "        }\n";

        // cut case
        fout << "        case_cut: {\n";
        graphs[ num_graph ].write_code( fout, "            " );
        fout << "            continue;\n";
        fout << "        }\n";
        fout << "    }\n";

        // store px, py, pi, ...
        for( char c : std::string( "xy" ) )
            for( std::size_t i = 0; i < outside.size() + 1 /*we may have to create a new point*/; i += simd_size )
                fout << "    SimdVec<TF," << simd_size << ">::store_aligned( p" << c << " + " << i << ", p" << c << "_" << i / simd_size << " );\n";

        fout << "}\n";
    }
    fout << "\n";
    fout << "int main( int argc, char **argv ) {\n";
    fout << "    bench_Cp2Lt64_code( { ";
    for( std::size_t num_graph = 0; num_graph < graphs.size(); ++num_graph )
        fout << ( num_graph ? ", " : "" ) << "cut_bench_" << num_graph;
    fout << " }, " << outside.size() << ", argc >= 2 ? argv[ 1 ] : nullptr );\n";
    fout << "}\n";
    fout << "\n";

    // launch the benchmark
    fout.close();

    cmd( "cd ~/sdot_pub && g++ -g3 -O3 -march=native -ffast-math -o .tmp.exe .tmp.cpp" );
    cmd( "./.tmp.exe .tmp.dat" );

    // get best graph
    std::ifstream fin( ".tmp.dat" );
    std::size_t best_gr;
    double score;
    fin >> best_gr
        >> score;

    res = graphs[ best_gr ];
    res.score = score;
    return true;
}


int main() {
    int max_simd_size = 8;
    int max_nb_nodes = 4;

    // ponderation
    std::vector<double> p_nb_nodes = { 0, 0, 0, 3, 20, 30, 25, 12, 5, 2 };
    std::vector<double> p_nb_cuts = { 65, 10, 10, 5, 1, 1, 1, 1, 1, 1 };

    std::map<unsigned,SimdGraph> graphs[ max_simd_size + 1 ];
    std::vector<double> scores( max_simd_size + 1, 0 );
    for( int nb_nodes = 3; nb_nodes <= max_nb_nodes; ++nb_nodes ) {
        for( unsigned comb = 0; comb < ( 1 << nb_nodes ); ++comb ) {
            for( int simd_size : { 1, 2, 4 } ) {
                SimdGraph gr;
                unsigned code = comb | ( 1 << nb_nodes );
                bool ok = make_code( gr, code, simd_size, max_nb_nodes );
                if ( ! ok )
                    continue;

                int nb_cuts = 0;
                for( int n = 0; n < nb_nodes; ++n )
                    nb_cuts += bool( comb & ( 1 << n ) );

                scores[ simd_size ] += p_nb_nodes[ nb_nodes ] * p_nb_cuts[ nb_cuts ] * gr.score;
                graphs[ simd_size ][ code ] = gr;
            }
        }
    }
    P( scores[ 1 ] );
    P( scores[ 2 ] );
    P( scores[ 4 ] );
}
