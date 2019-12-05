#include <iostream>
#include <sstream>
#include <bitset>
#include <vector>
#include <map>
#include <set>

void make_case( std::ostream &os, unsigned nb_nodes, std::bitset<8> outside_nodes ) {
    bool all_outside = true;
    bool all_inside = true;
    for( unsigned i = 0; i < nb_nodes; ++i ) {
        all_outside &= outside_nodes[ i ] == 1;
        all_inside &= outside_nodes[ i ] == 0;
    }

    // fully inside (should not appear at this point)
    if ( all_inside ) {
        os << "    // all_inside\n";
        return;
    }

    // fully outside
    if ( all_outside ) {
        os << "    // remove face\n";
        os << "    faces_to_rem[ nb_faces_to_rem++ ] = num_face;\n";
        os << "    faces.node_masks[ num_face ] = 0;\n";
        return;
    }

    // needed num nodes
    std::vector<std::size_t> off_out_ins;
    std::vector<std::size_t> off_in_outs;
    std::set<std::size_t> needed_num_nodes;
    for( std::size_t i = 0; i < nb_nodes; ++i ) {
        std::size_t n0 = ( i + 0 ) % nb_nodes;
        std::size_t n1 = ( i + 1 ) % nb_nodes;
        if ( outside_nodes[ n0 ] != outside_nodes[ n1 ] ) {
            needed_num_nodes.insert( n0 );
            needed_num_nodes.insert( n1 );

            if ( outside_nodes[ n0 ] )
                off_out_ins.push_back( i );
            if ( outside_nodes[ n1 ] )
                off_in_outs.push_back( i );
        }
    }
    if ( off_in_outs.size() > 1 || off_out_ins.size() > 1 )
        os << "    TODO; // several in_outs ou out_ins\n";
    for( std::size_t num_node : needed_num_nodes )
        os << "    int num_node_" << num_node << " = faces.node_lists[ num_face ][ " << num_node << " ];\n";

    // new nodes
    unsigned add_node_size = nb_nodes;
    std::vector<std::size_t> final_num_nodes;
    for( std::size_t i = 0; i < nb_nodes; ++i ) {
        std::size_t n0 = ( i + off_in_outs[ 0 ] + 0 ) % nb_nodes;
        std::size_t n1 = ( i + off_in_outs[ 0 ] + 1 ) % nb_nodes;

        // edges
        if ( outside_nodes[ n0 ] != outside_nodes[ n1 ] ) {
            std::string edge = std::to_string( n0 ) + "_" + std::to_string( n1 );
            os << "    std::size_t min_node_" << edge << " = min( num_node_" << n0 << ", num_node_" << n1 << " );\n";
            os << "    std::size_t max_node_" << edge << " = max( num_node_" << n0 << ", num_node_" << n1 << " );\n";
            os << "    std::size_t num_edge_" << edge << " = max_node_" << edge << " * ( max_node_" << edge << " + 1 ) / 2 + min_node_" << edge << ";\n";
            os << "    if ( edge_num_cuts[ num_edge_" << edge << " ] != num_cut ) {\n";
            os << "        edge_num_cuts[ num_edge_" << edge<< " ] = num_cut;\n";
            os << "        Node *node = &nodes.local_at( " << add_node_size << " );\n";
            os << "        edge_cuts[ num_edge_" << edge << " ] = node;\n";
            os << "        \n";
            os << "        const Node &n0 = nodes.local_at( num_node_" << n0 << " );\n";
            os << "        const Node &n1 = nodes.local_at( num_node_" << n1 << " );\n";
            os << "        node->set_pos( n0.pos() + n0.d / ( n0.d - n1.d ) * ( n1.pos() - n0.pos() ) );\n";
            os << "    }\n";
        }

        // num nodes
        if ( outside_nodes[ n0 ] == 0 ) {
            final_num_nodes.push_back( n0 );
            if ( outside_nodes[ n1 ] )
                final_num_nodes.push_back( n0 );
        } else if ( outside_nodes[ n1 ] == 0 ) {

        }
    }

    // vec of num nodes
    //    for( std::size_t i = 0; i < nb_nodes; ++i ) {
    //        std::size_t n0 = ( i + off_in_outs[ 0 ] + 0 ) % nb_nodes;
    //        std::size_t n1 = ( i + off_in_outs[ 0 ] + 0 ) % nb_nodes;
    //    }

}

int main() {
    // code content => code number
    std::map<std::string,unsigned> code_map;
    code_map[ "" ] = 0;

    // cases num => code number
    std::vector<unsigned> case_nums( ( ( 1 << 8 ) << 3 ) + 8, 0 ); // case => code number
    for( unsigned nb_nodes = 3; nb_nodes <= 8; ++nb_nodes ) {
        for( unsigned outside_case = 0; outside_case < ( 1 << 8 ); ++outside_case ) {
            // make the code
            std::ostringstream os;
            make_case( os, nb_nodes, outside_case );

            // get number
            auto iter = code_map.find( os.str() );
            if ( iter == code_map.end() )
                iter = code_map.insert( iter, { os.str(), code_map.size() } );

            // store the case
            unsigned code_val = ( outside_case << 3 ) + nb_nodes - 3;
            case_nums[ code_val ] = iter->second;
        }
    }

    // jump code
    std::cout << "static void *dispatch_table[] = {\n";
    for( unsigned case_num : case_nums )
        std::cout << "    &&case_" << case_num << ",\n";
    std::cout << "};\n";
    std::cout << "using std::min;\n";
    std::cout << "using std::max;\n";
    std::cout << "goto *dispatch_table[ ouf ];\n";

    // cases code
    for( auto iter : code_map ) {
        if ( iter.second == 0 )
            continue;
        std::cout << "case_" << iter.second << ": {\n";
        std::cout << iter.first;
        std::cout << "    break;\n";
        std::cout << "}\n";
    }
    std::cout << "case_0:\n";
    std::cout << "    ;\n";
}
