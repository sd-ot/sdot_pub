#include "../Support/Display/DotOut.h"
#include "SimdGraph.h"
#include <fstream>

SimdGraph::SimdGraph() {
    cur_op_id = 0;
}

void SimdGraph::for_each_child( const std::function<void(SimdOp *)> &f, const std::vector<SimdOp *> &targets ) {
    ++cur_op_id;
    for( SimdOp *target : targets )
        for_each_child_rec( f, target );

}

void SimdGraph::add_target( SimdOp *target ) {
    targets.push_back( target );
}

SimdOp *SimdGraph::make_op( std::string name, const std::vector<SimdOp *> &children ) {
    pool.emplace_back( name, children );
    return &pool.back();
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

void SimdGraph::for_each_child_rec( const std::function<void (SimdOp *)> &f, SimdOp *target ) {
    if ( target->op_id == cur_op_id )
        return;
    target->op_id = cur_op_id;

    f( target );

    for( SimdOp *ch : target->children )
        for_each_child_rec( f, ch );
}
