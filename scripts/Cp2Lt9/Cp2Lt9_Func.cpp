#include "../../src/sdot/Support/TODO.h"
#include "../../src/sdot/Support/P.h"
#include "Cp2Lt9_Func.h"
#include <sstream>
#include <fstream>

Cp2Lt9_Func::Cp2Lt9_Func( OptParm &opt_parm, std::string float_type, std::string simd_type, int max_nb_nodes ) : max_nb_nodes( max_nb_nodes ), float_type( float_type ), simd_type( simd_type ) {
    // func parms
    size_for_tests = 4;
    min_nb_nodes = 3;
    nb_registers = 6;
    pi_in_regs = 0;
    simd_size = 1 << opt_parm.get_value( max_log_simd_size() );
    make_di = opt_parm.get_value( 2 );

    // find best code for each comb nb_nodes/outside_nodes
    make_case_map();
}

Cp2Lt9_Func::Cp2Lt9_Func() {
}

int Cp2Lt9_Func::max_log_simd_size() const {
    if ( simd_type == "SSE2"   ) return float_type == "float" ? 2 : 1;
    if ( simd_type == "AVX2"   ) return float_type == "float" ? 3 : 2;
    if ( simd_type == "AVX512" ) return float_type == "float" ? 4 : 3;
    return 0;
}

void Cp2Lt9_Func::write_def( std::ostream &os ) const {
    write_def( os, case_map, "ConvexPolyhedron2d_Lt9cut" );
}

double Cp2Lt9_Func::score() const {
    if ( case_map.empty() )
        return std::numeric_limits<double>::max();

    // weighting
    std::vector<double> p_nb_nodes = { 0, 0, 0, 3, 20, 30, 25, 12, 5, 2, 1 };
    std::vector<double> p_nb_cuts = { 65, 10, 10, 5, 1, 1, 1, 1, 1, 1, 1 };

    double res = 0;
    for( int nb_nodes = 3; nb_nodes <= max_nb_nodes; ++nb_nodes ) {
        for( unsigned comb = 0; comb < ( 1 << nb_nodes ); ++comb ) {
            int nb_cuts = 0;
            for( int n = 0; n < nb_nodes; ++n )
                nb_cuts += bool( comb & ( 1 << n ) );

            unsigned code = comb | ( 1 << nb_nodes );
            res += case_map.find( code )->second.score * p_nb_nodes[ nb_nodes ] * p_nb_cuts[ nb_cuts ];
        }
    }

    return res;
}

void Cp2Lt9_Func::make_best_score( double &best_score, std::size_t &best_case, const std::vector<Cp2Lt9_Case> &cases, unsigned code, int nb_nodes ) {
    std::ofstream os( ".tmp.cpp" );
    os << "#include \"scripts/Cp2Lt9/Cp2Lt9_bench.h\"\n";
    os << "\n";

    // write functions to be tested
    for( std::size_t num_case = 0; num_case < cases.size(); ++num_case ) {
        CaseMap case_map;
        case_map[ code ] = { cases[ num_case ], 0.0 };

        //    os << "        case_init: {\n";
        //    os << "            nodes_size = " << outside.size() << ";\n";
        //    for( std::size_t j = 0; j < outside.size(); ++j )
        //        os << "            px_" << j / simd_size << "[ " << j % simd_size << " ] = " << ( outside[ j ] ? 1 : -1 ) << ";\n";
        //    os << "            continue;\n";
        //    os << "        }\n";

        os << "\n";
        write_def( os, case_map, "cut_bench_" + std::to_string( num_case ), true );
    }

    // write main
    os << "\n";
    os << "int main( int argc, char **argv ) {\n";
    os << "    using TF = " << ( float_type == "gen" ? "double" : float_type ) << ";\n";
    os << "    using TC = std::size_t;\n";
    os << "    using FU = std::function<void( TF *px, TF *py, TC *pi, int &nodes_size, const TF *cut_x, const TF *cut_y, const TF *cut_s, const TC *cut_i, int cut_n )>;\n";
    os << "    bench_Cp2Lt64_code( std::vector<FU>{ ";
    for( std::size_t num_case = 0; num_case < cases.size(); ++num_case )
        os << ( num_case ? ", " : "" ) << "cut_bench_" << num_case;
    os << " }, " << nb_nodes << ", argc >= 2 ? argv[ 1 ] : nullptr );\n";
    os << "}\n";
    os << "\n";

    // compile and execute
    os.close();

    //    cmd( "clang++ -O3" + ( arch.empty() ? "" : " -march=" + arch ) + " -ffast-math -o .tmp.exe .tmp.cpp" );
    //    cmd( "./.tmp.exe .tmp.dat" );

    //    // get best graph
    //    std::ifstream fin( ".tmp.dat" );
    //    std::size_t best_gr;
    //    double score;
    //    fin >> best_gr
    //        >> score;
}

void Cp2Lt9_Func::make_case_map() {
    for( int nb_nodes = min_nb_nodes; nb_nodes <= max_nb_nodes; ++nb_nodes ) {
        for( unsigned comb = 0; comb < ( 1 << nb_nodes ); ++comb ) {
            // find all possible way to make the case
            std::vector<Cp2Lt9_Case> cases;
            OptParm loc_opt_parm;
            do {
                Cp2Lt9_Case ca( loc_opt_parm, nb_nodes, comb, simd_size, nb_registers );
                if ( ca.valid )
                    cases.push_back( ca );
            } while ( loc_opt_parm.inc() );

            // write a .cpp file that compute the timing for each proposition
            if ( cases.size() ) {
                double best_score;
                std::size_t best_case = 0;
                unsigned code = comb | ( 1 << nb_nodes );
                make_best_score( best_score, best_case, cases, code, nb_nodes );

                // store the best one
                case_map[ code ] = { cases[ best_case ], best_score };
            }
        }
    }
}

void Cp2Lt9_Func::write_def( std::ostream &os, const CaseMap &case_map, std::string func_name, bool /*for_1_case*/ ) const {
    os << "// simd_size: " << simd_size << " make_di: " << make_di << " \n";

    // function signature
    if ( float_type == "gen" )
        os << "template<class TF,class TC>\n";
    os << "bool " << func_name << "( TF *px, TF *py, TC *pi, int &nodes_size, const TF *cut_x, const TF *cut_y, const TF *cut_s, const TC *cut_i, int cut_n ) {\n";
    os << "    using VF = SimdVec<TF," << simd_size << ">;\n";
    os << "    using VC = SimdVec<TC," << simd_size << ">;\n";

    // decl of registers (px, py, pi)
    os << "\n";
    for( char c : std::string( "xy" ) ) {
        os << "    VF ";
        for( int i = 0; i < nb_registers; ++i )
            os << ( i ? ", " : "" ) << "p" << c << "_" << i;
        os << ";\n";
    }
    if ( pi_in_regs ) {
        os << "    VC ";
        for( int i = 0; i < nb_registers; ++i )
            os << ( i ? ", " : "" ) << "pi_" << i;
        os << ";\n";
    }

    // load registers
    os << "\n";
    for( int i = 0, sp = 4; ; ++i ) {
        if ( i == nb_registers ) {
            while( --i * simd_size > size_for_tests )
                os << std::string( sp -= 4, ' ' ) << "}\n";
            break;
        }

        if ( i * simd_size > size_for_tests ) {
            os << std::string( sp, ' ' ) << "if ( nodes_size >= " << i * simd_size << " ) {\n";
            sp += 4;
        }

        for( char c : std::string( "xy" ) )
            os << std::string( sp, ' ' ) << "p" << c << "_" << i << " = VF::load_aligned( p" << c << " + " << simd_size * i << " );\n";
        if ( pi_in_regs )
            os << std::string( sp, ' ' ) << "pi_" << i << " = VC::load_aligned( pi + " << simd_size * i << " );\n";
    }


    // function to store px, py, pi, ...
    os << "\n";
    os << "    auto store = [&]() {\n";
    for( int i = 0, sp = 8; ; ++i ) {
        if ( i == nb_registers ) {
            while( --i * simd_size > size_for_tests )
                os << std::string( sp -= 4, ' ' ) << "}\n";
            break;
        }

        if ( i * simd_size > size_for_tests ) {
            os << std::string( sp, ' ' ) << "if ( nodes_size >= " << i * simd_size << " ) {\n";
            sp += 4;
        }

        for( char c : std::string( "xy" ) )
            os << std::string( sp, ' ' ) << "VF::store_aligned( p" << c << " + " << simd_size * i << ", p" << c << "_" << i << " );\n";
        if ( pi_in_regs )
            os << std::string( sp, ' ' ) << "VF::store_aligned( pi + " << simd_size * i << ", pi_" << i << " );\n";
    }
    os << "    };\n";

    // loop over the cuts
    os << "\n";
    os << "    for( int num_cut = 0; ; ++num_cut ) {\n";
    os << "        if ( num_cut == cut_n ) {\n";
    os << "            store();\n";
    os << "            return true;\n";
    os << "        }\n";

    // get distance and outside bit for each node
    os << "\n";
    os << "        VC ci = cut_i[ num_cut ];\n";
    os << "        VF cx = cut_x[ num_cut ];\n";
    os << "        VF cy = cut_y[ num_cut ];\n";
    os << "        VF cs = cut_s[ num_cut ];\n";
    os << "        \n";
    os << "        int outside_nodes = 0;\n";
    os << "        int nmsk = 1 << nodes_size;\n";

    //
    auto val_reg = [&]( std::string c, int num_reg ) {
        if ( num_reg < nb_registers )
            return "p" + c + "_" + std::to_string( num_reg );
        return std::string( c == "i" ? "VC" : "VF" ) + "::load_aligned( p" + c + " + " + std::to_string( num_reg * simd_size ) + " )";
    };

    if ( make_di ) {
        for( int i = 0; i < max_nb_nodes; i += simd_size )
            os << "        VF di_" << i / simd_size << " = " << val_reg( "x", i / simd_size ) << " * cx + " << val_reg( "y", i / simd_size ) << " * cy - cs; outside_nodes |= ( di_" << i / simd_size << " > 0 ) << " << i << ";\n";
    } else {
        for( int i = 0; i < max_nb_nodes; i += simd_size )
            os << "        VF bi_" << i / simd_size << " = " << val_reg( "x", i / simd_size ) << " * cx + " << val_reg( "x", i / simd_size ) << " * cy; outside_nodes |= ( bi_" << i / simd_size << " > cs ) << " << i << "; VF di_" << i / simd_size << " = bi_" << i / simd_size << " - cs;\n";
    }

    os << "        \n";
    os << "        // if nothing has changed => go to the next cut\n";
    os << "        int case_code = ( outside_nodes & ( nmsk - 1 ) ) | nmsk;\n";
    os << "        if ( outside_nodes == 0 )\n";
    os << "            continue;\n";

    // dispatch
    std::map<std::string,std::size_t> code_map;
    os << "        static void *dispatch_table[] = {";
    for( unsigned code = 0; code < ( 1 << ( max_nb_nodes + 1 ) ); ++code ) {
        if ( code % 16 == 0 )
            os << "\n            ";

        if ( case_map.count( code ) == 0 ) {
            os << "&&case_nhdl, ";
            continue;
        }

        std::ostringstream fcc;
        fcc << case_map.find( code )->second.code.code;
        if ( code_map.count( fcc.str() ) == 0 )
            code_map[ fcc.str() ] = code_map.size();
        os << "&&case_" << std::setw( 4 ) << std::left << code_map[ fcc.str() ] << ", ";
    }
    os << "\n        };\n";

    os << "        \n";
    os << "        goto *dispatch_table[ case_code ];\n";

    // not handled
    os << "        \n";
    os << "        case_nhdl: {\n";
    os << "            store();\n";
    os << "            return false;\n";
    os << "        }\n";

    // cut case
    for( std::pair<std::string,std::size_t> c : code_map ) {
        os << "        \n";
        os << "        case_" << c.second << ": {\n";
        os << c.first;
        os << "        }\n";
    }

    os << "    }\n";
    os << "}\n";
}

//#include "../src/sdot/SimdCodegen/SimdGraph.h"
//#include "../src/sdot/Support/OptParm.h"
//#include "../src/sdot/Support/ASSERT.h"
//#include "../src/sdot/Support/ERROR.h"

//static void disp_sp( std::ostream &os, std::string sp, std::string str ) {
//    if ( str.empty() )
//        return;
//    bool need_sp = true;
//    for( char c : str ) {
//        if ( c == '\n' )
//            need_sp = true;
//        else if ( need_sp ) {
//            need_sp = false;
//            os << sp;
//        }
//        os << c;
//    }
//}

//struct CodeGraph {
//    void        write_code( std::ostream &os, std::string sp ) { disp_sp( os, sp, prel ); graph.write_code( os, sp ); disp_sp( os, sp, suff ); }
//    bool        make_di;
//    SimdGraph   graph;
//    double      score;
//    std::string prel;
//    std::string suff;
//};

//void cmd( std::string c ) {
//    std::cout << c << std::endl;
//    if ( system( c.c_str() ) )
//        ERROR( "..." );
//}

//CodeGraph make_graph( OptParm &opt_parm, const Cp2Lt64CutList &mod, std::vector<bool> outside, int simd_size, bool make_di ) {
//    std::vector<std::size_t> sp_ind = mod.split_indices();
//    ASSERT( sp_ind.size() == 2, "" );
//    CodeGraph res;
//    res.make_di = make_di;

//    if ( outside.size() != mod.ops.size() )
//        res.prel += "nodes_size = " + std::to_string( mod.ops.size() ) + ";\n";

//    int n0 = mod.ops[ sp_ind[ 0 ] ].n0(), n1 = mod.ops[ sp_ind[ 0 ] ].n1();
//    int n2 = mod.ops[ sp_ind[ 1 ] ].n0(), n3 = mod.ops[ sp_ind[ 1 ] ].n1();
//    bool switch_cuts = opt_parm.get_value( 2 );
//    if ( switch_cuts ) {
//        std::swap( n0, n2 );
//        std::swap( n1, n3 );
//    }

//    SimdGraph &gr = res.graph;
//    std::vector<SimdOp *> px_s, py_s, pi_s, di_s;
//    for( std::size_t i = 0; i < outside.size(); i += simd_size ) {
//        px_s.push_back( gr.make_op( "REG px_" + std::to_string( i / simd_size ) + " d " + std::to_string( simd_size ), {} ) );
//        py_s.push_back( gr.make_op( "REG py_" + std::to_string( i / simd_size ) + " d " + std::to_string( simd_size ), {} ) );
//        pi_s.push_back( gr.make_op( "REG pi_" + std::to_string( i / simd_size ) + " i " + std::to_string( simd_size ), {} ) );
//        di_s.push_back( gr.make_op( "REG di_" + std::to_string( i / simd_size ) + " d " + std::to_string( simd_size ), {} ) );
//    }

//    SimdOp *px_a = gr.make_op( "AGG", { gr.get_op( px_s[ n0 / simd_size ], n0 % simd_size ), gr.get_op( px_s[ n2 / simd_size ], n2 % simd_size ) } );
//    SimdOp *px_b = gr.make_op( "AGG", { gr.get_op( px_s[ n1 / simd_size ], n1 % simd_size ), gr.get_op( px_s[ n3 / simd_size ], n3 % simd_size ) } );
//    SimdOp *py_a = gr.make_op( "AGG", { gr.get_op( py_s[ n0 / simd_size ], n0 % simd_size ), gr.get_op( py_s[ n2 / simd_size ], n2 % simd_size ) } );
//    SimdOp *py_b = gr.make_op( "AGG", { gr.get_op( py_s[ n1 / simd_size ], n1 % simd_size ), gr.get_op( py_s[ n3 / simd_size ], n3 % simd_size ) } );
//    SimdOp *di_a = gr.make_op( "AGG", { gr.get_op( di_s[ n0 / simd_size ], n0 % simd_size ), gr.get_op( di_s[ n2 / simd_size ], n2 % simd_size ) } );
//    SimdOp *di_b = gr.make_op( "AGG", { gr.get_op( di_s[ n1 / simd_size ], n1 % simd_size ), gr.get_op( di_s[ n3 / simd_size ], n3 % simd_size ) } );

//    SimdOp *di_m = gr.make_op( "DIV", { di_a, gr.make_op( "SUB", { di_b, di_a } ) } );
//    SimdOp *addx = gr.make_op( "ADD", { px_a, gr.make_op( "MUL", { di_m, gr.make_op( "SUB", { px_a, px_b } ) } ) } );
//    SimdOp *addy = gr.make_op( "ADD", { py_a, gr.make_op( "MUL", { di_m, gr.make_op( "SUB", { py_a, py_b } ) } ) } );

//    std::vector<std::vector<SimdOp *>> x_ch( ( mod.ops.size() + simd_size - 1 ) / simd_size );
//    std::vector<std::vector<SimdOp *>> y_ch( ( mod.ops.size() + simd_size - 1 ) / simd_size );
//    std::vector<std::vector<SimdOp *>> i_ch( ( mod.ops.size() + simd_size - 1 ) / simd_size );
//    int num_in_adds = switch_cuts;
//    for( std::size_t i = 0; i < mod.ops.size(); ++i ) {
//        if ( mod.ops[ i ].single() ) {
//            x_ch[ i / simd_size ].push_back( gr.get_op( px_s[ mod.ops[ i ].i0 / simd_size ], mod.ops[ i ].i0 % simd_size ) );
//            y_ch[ i / simd_size ].push_back( gr.get_op( py_s[ mod.ops[ i ].i0 / simd_size ], mod.ops[ i ].i0 % simd_size ) );
//            i_ch[ i / simd_size ].push_back( gr.get_op( pi_s[ mod.ops[ i ].i0 / simd_size ], mod.ops[ i ].i0 % simd_size ) );
//        } else {
//            x_ch[ i / simd_size ].push_back( gr.get_op( addx, num_in_adds ) );
//            y_ch[ i / simd_size ].push_back( gr.get_op( addy, num_in_adds ) );
//            i_ch[ i / simd_size ].push_back( gr.make_op( "REG cut_i[num_cut] i 1", {} ) );
//            num_in_adds ^= 1;
//        }
//    }

//    for( std::size_t i = 0; i < x_ch.size(); ++i ) {
//        while ( int( x_ch[ i ].size() ) < simd_size ) x_ch[ i ].push_back( x_ch[ i ].back() );
//        while ( int( y_ch[ i ].size() ) < simd_size ) y_ch[ i ].push_back( y_ch[ i ].back() );
//        while ( int( i_ch[ i ].size() ) < simd_size ) i_ch[ i ].push_back( i_ch[ i ].back() );
//        gr.add_target( gr.make_op( "SET px_" + std::to_string( i ), { gr.make_op( "AGG"   , x_ch[ i ] ) } ) );
//        gr.add_target( gr.make_op( "SET py_" + std::to_string( i ), { gr.make_op( "AGG"   , y_ch[ i ] ) } ) );
//        gr.add_target( gr.make_op( "SET pi_" + std::to_string( i ), { gr.make_op( "AGG TC", i_ch[ i ] ) } ) );
//    }

//    res.prel += va_string( "// mod={}, swith_cuts={}\n", mod, switch_cuts );
//    res.suff += "continue;\n";
//    // res.gr.display();
//    return res;
//}

//        // init case
//        os << "        case_init: {\n";
//        os << "            nodes_size = " << outside.size() << ";\n";
//        for( std::size_t j = 0; j < outside.size(); ++j )
//            os << "            px_" << j / simd_size << "[ " << j % simd_size << " ] = " << ( outside[ j ] ? 1 : -1 ) << ";\n";
//        os << "            continue;\n";
//        os << "        }\n";
//void write_for( std::string float_type, std::string simd_type, int max_nb_nodes ) {
//    int max_simd_size = 1;
//    std::string arch, simd_test;
//    if ( simd_type == "SSE2"   ) { max_simd_size = float_type == "float" ?  4 : 2; arch = "skylake"       ; simd_test = "__SSE2__"   ; }
//    if ( simd_type == "AVX2"   ) { max_simd_size = float_type == "float" ?  8 : 4; arch = "skylake"       ; simd_test = "__AVX2__"   ; }
//    if ( simd_type == "AVX512" ) { max_simd_size = float_type == "float" ? 16 : 8; arch = "skylake-avx512"; simd_test = "__AVX512F__"; }

//    // declaration
//    if ( float_type != "gen" ) {
//        std::string nh = va_string( "src/sdot/ConvexPolyhedron/Internal/ConvexPolyhedron2dLt64_cut_{}_{}.h" , float_type, simd_type );
//        std::ofstream fh( nh.c_str() );
//        if ( ! simd_test.empty() )
//            fh << "#ifdef " << simd_test << "\n";
//        fh << "namespace sdot {\n";
//        fh << "namespace internal {\n";
//        fh << "bool ConvexPolyhedron2dLt64_cut( int &num_cut, " << float_type << " *px, " << float_type << " *py, std::size_t *pi, int &nodes_size, const " << float_type << " *cut_x, const " << float_type << " *cut_y, const " << float_type << " *cut_s, const std::size_t *cut_i, int cut_n );\n";
//        fh << "} // namespace internal\n";
//        fh << "} // namespace sdot\n";
//        if ( ! simd_test.empty() )
//            fh << "#endif // " << simd_test << "\n";
//    }

//    // definition
//    std::string nc = va_string( "src/sdot/ConvexPolyhedron/Internal/ConvexPolyhedron2dLt64_cut_{}_{}.{}", float_type, simd_type, float_type == "gen" ? "h" : "cpp" );
//    std::ofstream fc( nc.c_str() );
//    if ( ! simd_test.empty() )
//        os << "#ifdef " << simd_test << "\n";
//    os << "#include \"../../Support/SimdVec.h\"\n";
//    os << "namespace sdot {\n";
//    os << "namespace internal {\n";
//    std::string _float_type = float_type;
//    if ( float_type == "gen" ) {
//        _float_type = "TF";
//        os << "template<class TF>\n";
//    }
//    os << "bool ConvexPolyhedron2dLt64_cut( int &num_cut, " << _float_type << " *px, " << _float_type << " *py, std::size_t *pi, int &nodes_size, const " << _float_type << " *cut_x, const " << _float_type << " *cut_y, const " << _float_type << " *cut_s, const std::size_t *cut_i, int cut_n ) {\n";
//    os << "    using namespace sdot;\n";
//    if ( float_type != "gen" )
//        os << "    using TF = " << float_type << ";\n";
//    os << "    using TC = std::size_t;\n";
//    os << "    using VF = SimdVec<TF," << simd_size << ">;\n";
//    os << "    using VC = SimdVec<TC," << simd_size << ">;\n";

