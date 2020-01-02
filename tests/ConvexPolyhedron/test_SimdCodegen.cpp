#include "../../src/sdot/SimdCodegen/SimdCodegen.h"
#include "../../src/sdot/Support/OptParm.h"
#include "../../src/sdot/Support/ASSERT.h"
#include "../../src/sdot/Support/P.h"

struct Op {
    /**/        Op             ( std::size_t i0 = 0, std::size_t i1 = 0, int dir = 0 ) : dir( dir ), i0( i0 ), i1( i1 ), sw( false ) {}

    void        write_to_stream( std::ostream &os ) const { if ( single() ) os << inside_node(); else os << "[" << i0 << "," << i1 << "]"; }
    bool        going_outside  () const { return dir  < 0; }
    bool        going_inside   () const { return dir  > 0; }
    bool        single         () const { return dir == 0; }
    bool        split          () const { return dir != 0; }

    std::size_t outside_node   () const { return dir > 0 ? i0 : i1; }
    std::size_t inside_node    () const { return dir > 0 ? i1 : i0; }

    std::size_t n0             () const { return sw ? i1 : i0; }
    std::size_t n1             () const { return sw ? i0 : i1; }

    int         dir;           ///< -1 => going outside. 0 => single node. +1 => going inside.
    std::size_t i0;            ///<
    std::size_t i1;            ///<
    bool        sw;            ///<
};

struct Mod {
    void                     write_to_stream( std::ostream &os ) const { os << ops; }
    std::vector<std::size_t> split_indices  () const { std::vector<std::size_t> res; for( std::size_t i = 0; i < ops.size(); ++i ) if ( ops[ i ].split() ) res.push_back( i ); return res; }
    void                     rotate         ( std::size_t off ) { std::vector<Op> nops( ops.size() ); for( std::size_t i = 0; i < ops.size(); ++i ) nops[ i ] = ops[ ( i + off ) % ops.size() ]; ops = nops; }
    void                     sw             ( std::uint64_t val ) { std::vector<std::size_t> si = split_indices(); for( std::size_t i = 0; i < si.size(); ++i ) ops[ si[ i ] ].sw = val & ( std::uint64_t( 1 ) << i ); }
    // double                score          ( std::string variant, int simd_size, int nb_regs );
    // void                  write          ( std::ostream &os, std::string variant, int simd_size, int nb_regs, std::string sp = "        " );

    std::vector<Op>          ops;
};

void make_graph( SimdCodegen &sc, OptParm &opt_parm, const Mod &mod ) {
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
    for( const Op &op : mod.ops ) {
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

    sc.add_possibility( gr );
}

bool make_code( std::ostream &os, std::vector<bool> outside ) {
    // make a ref Mod
    Mod ref_mod;
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

    //
    SimdCodegen sc;
    OptParm opt_parm;
    do {
        Mod mod = ref_mod;
        mod.rotate( opt_parm.get_value( mod.ops.size() ) );
        mod.sw( opt_parm.get_value( 1 << mod.split_indices().size() ) );

        make_graph( sc, opt_parm, mod );
    } while ( opt_parm.inc() );

    sc.write_code( std::cout );
    return true;
}

int main( int /*argc*/, char **/*argv*/ ) {
    make_code( std::cout, { 0, 1, 1, 0 } );
}
