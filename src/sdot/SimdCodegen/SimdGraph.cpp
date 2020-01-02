#include "../Support/Display/DotOut.h"
#include "../Support/TODO.h"
#include "SimdGraph.h"
#include <fstream>

SimdGraph::SimdGraph( const SimdGraph &that ) {
   operator=( that );
}

SimdGraph::SimdGraph() {
    cur_op_id = 0;
}

void SimdGraph::operator=( const SimdGraph &that ) {
    this->cur_op_id = that.cur_op_id;
    this->msg = that.msg;

    pool.clear();

    for( const SimdOp &op : that.pool ) {
        pool.push_back( { op.name, op.children, op.op_id } );
        op.repl = &pool.back();
    }

    for( const SimdOp *op : that.targets )
        targets.push_back( op->repl );

    for( SimdOp &op : pool )
        for( SimdOp *&ch : op.children )
            ch = ch->repl;
}

void SimdGraph::for_each_child( const std::function<void(SimdOp *)> &f, const std::vector<SimdOp *> &targets, bool postfix ) const {
    ++cur_op_id;
    for( SimdOp *target : targets )
        for_each_child_rec( f, target, postfix );

}

void SimdGraph::add_target( SimdOp *target ) {
    targets.push_back( target );
}

void SimdGraph::set_msg( std::string msg ) {
    this->msg = msg;
}

void SimdGraph::write_code( std::ostream &os, std::string sp ) {
    if ( msg.size() )
        os << sp << "// " << msg << "\n";

    // update parents
    std::vector<SimdOp *> front;
    update_parents( &front );

    //
    ++cur_op_id;
    int nb_regs = 0;
    while ( front.size() ) {
        for( std::size_t i = 0; i < front.size() - 1; ++i )
            if ( front[ i ]->better_code( front.back() ) )
                std::swap( front[ i ], front.back() );
        SimdOp *op = front.back();
        front.pop_back();

        if ( op->op_id == cur_op_id )
            continue;
        op->op_id = cur_op_id;

        op->write_code( os, sp, nb_regs );

        auto all_children_done = [&]( SimdOp *pa ) {
            for( SimdOp *ch : pa->children )
                if ( ch->op_id != cur_op_id )
                    return false;
            return true;
        };
        for( SimdOp *pa : op->parents )
            if ( all_children_done( pa ) )
                front.push_back( pa );
    }
}

SimdOp *SimdGraph::make_op( std::string name, const std::vector<SimdOp *> &children ) {
    for( SimdOp &op : pool )
        if ( op.name == name && op.children == children )
            return &op;
    pool.emplace_back( name, children );
    return &pool.back();
}

SimdOp *SimdGraph::get_op( SimdOp *op, int num ) {
    return make_op( "GET " + std::to_string( num ), { op } );
}

void SimdGraph::display( std::string filename ) {
    std::ofstream os( filename.c_str() );

    os << "digraph LexemMaker {\n";
    for_each_child( [&]( SimdOp *node ) {
        std::ostringstream ss_node;
        node->write_to_stream( ss_node );

        os << "  node_" << node << " [label=\"";
        dot_out( os, ss_node.str() );
        os << "\"];\n";

        for( std::size_t i = 0; i < node->children.size(); ++i ) {
            std::ostringstream ss_trans;
            // node->write_edge_info( ss_trans, i );
            // ss_trans << node->transitions[ i ].provenance;

            os << "  node_" << node << " -> node_" << node->children[ i ] << " [label=\"";
            dot_out( os, ss_trans.str() );
            os << "\"];\n";
        }
    }, targets );
    os << "}\n";

    os.close();

    exec_dot( filename.c_str() );
}

void SimdGraph::for_each_child_rec( const std::function<void (SimdOp *)> &f, SimdOp *target, bool postfix ) const {
    if ( target->op_id == cur_op_id )
        return;
    target->op_id = cur_op_id;

    if ( postfix == false )
        f( target );

    for( SimdOp *ch : target->children )
        for_each_child_rec( f, ch, postfix );

    if ( postfix )
        f( target );
}

void SimdGraph::update_parents( std::vector<SimdOp *> *front ) {
    for_each_child( [&]( SimdOp *op ) {
        op->parents.clear();
    }, targets );

    for_each_child( [&]( SimdOp *op ) {
        if ( op->children.empty() && front )
            front->push_back( op );
        else
            for( SimdOp *ch : op->children )
                ch->parents.push_back( op );
    }, targets );
}
