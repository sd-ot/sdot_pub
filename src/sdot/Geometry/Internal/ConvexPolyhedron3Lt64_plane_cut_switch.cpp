#include "../../Support/P.h"
#include <sstream>
#include <bitset>
#include <vector>
#include <map>
#include <set>

/// nothing in os => generic case
void make_case( std::ostream &os, int nb_nodes, std::bitset<8> outside_nodes ) {
    using std::min;
    using std::max;
    int nb_outside = 0;
    int nb_inside = 0;
    for( int i = 0; i < nb_nodes; ++i ) {
        nb_outside += outside_nodes[ i ] == 1;
        nb_inside += outside_nodes[ i ] == 0;
    }

    // fully inside (should not appear at this point)
    if ( nb_inside == nb_nodes ) {
        // we must put something to avoid the generic case
        os << "    // all_inside\n";
        return;
    }

    // fully outside
    if ( nb_inside == 0 ) {
        os << "    // remove face\n";
        os << "    faces_to_rem[ nb_faces_to_rem++ ] = num_face;\n";
        os << "    faces.node_masks[ num_face ] = 0; // to say that this face is free\n";
        return;
    }

    // if < 64 => num_node. Else, num_edge
    std::set<int> registered_num_nodes;
    auto get_node = [&]( int num_node ) {
        if ( num_node >= 64 || registered_num_nodes.count( num_node ) )
            return;
        registered_num_nodes.insert( num_node );
        os << "    int num_node_" << num_node << " = faces.node_lists[ num_face ][ " << num_node << " ];\n";
    };

    auto nedge = [&]( int n0, int n1 ) {
        get_node( n0 );
        get_node( n1 );
        int mi = min( n0, n1 );
        int ma = max( n0, n1 );
        return 64 + ma * ( ma + 1 ) / 2 + mi;
    };

    // make/find the new nodes (on in/out ot out/in edges)
    for( int i = 0; i < nb_nodes; ++i ) {
        int n0 = ( i + 0 ) % nb_nodes;
        int n1 = ( i + 1 ) % nb_nodes;
        if ( outside_nodes[ n0 ] != outside_nodes[ n1 ] ) {
            int edge = nedge( n0, n1 );
            os << "    int min_node_" << edge << " = min( num_node_" << n0 << ", num_node_" << n1 << " );\n";
            os << "    int max_node_" << edge << " = max( num_node_" << n0 << ", num_node_" << n1 << " );\n";
            os << "    int num_edge_" << edge << " = max_node_" << edge << " * ( max_node_" << edge << " + 1 ) / 2 + min_node_" << edge << ";\n";
            os << "    int num_node_" << edge << ";\n";
            os << "    if ( edge_num_cuts[ num_edge_" << edge << " ] != num_cut ) {\n";
            os << "        const Node &n0 = nodes.local_at( num_node_" << n0 << " );\n";
            os << "        const Node &n1 = nodes.local_at( num_node_" << n1 << " );\n";
            os << "        edge_num_cuts[ num_edge_" << edge << " ] = num_cut;\n";
            os << "        num_node_" << edge << " = new_nodes_size++;\n";
            os << "        edge_cuts[ num_edge_" << edge << " ] = num_node_" << edge<< ";\n";
            os << "        nodes.local_at( num_node_" << edge << " ).set_pos( n0.pos() + n0.d / ( n0.d - n1.d ) * ( n1.pos() - n0.pos() ) );\n";
            os << "    } else\n";
            os << "        num_node_" << edge << " = edge_cuts[ num_edge_" << edge << " ];\n";
        }
    }

    //
    int nb_in_outs = 0;
    int nb_out_ins = 0;
    std::vector<int> new_nodes;
    for( int i = 0; i < nb_nodes; ++i ) {
        int nm = ( i + nb_nodes - 1 ) % nb_nodes;
        int n0 = ( i +            0 ) % nb_nodes;
        int n1 = ( i +            1 ) % nb_nodes;
        if ( outside_nodes[ n0 ] )
            continue;

        if ( outside_nodes[ nm ] ) { // out/in
            new_nodes.push_back( nedge( nm, n0 ) );
            ++nb_out_ins;
        }

        new_nodes.push_back( n0 );

        if ( outside_nodes[ n1 ] ) { // in/out
            new_nodes.push_back( nedge( n0, n1 ) );
            ++nb_in_outs;
        }
    }

    if ( nb_out_ins >= 2 || nb_in_outs >= 2 ) {
        os.clear();
        return;
    }

    // write node indices
    os << "    // " << new_nodes << "\n";
    if ( int( new_nodes.size() ) != nb_nodes )
        os << "    faces.nb_nodes[ num_face ] = " << new_nodes.size() << ";\n";

    for( std::size_t i = 0; i < new_nodes.size(); ++i )
        if ( new_nodes[ i ] != int( i ) )
            get_node( new_nodes[ i ] );

    for( std::size_t i = 0; i < new_nodes.size(); ++i )
        if ( new_nodes[ i ] != int( i ) )
            os << "    faces.node_lists[ num_face ][ " << i << " ] = num_node_" << new_nodes[ i ] << ";\n";
}

int main() {
    // code content => code number. void string => generic case (case 0)
    std::map<std::string,unsigned> code_map;
    code_map[ "" ] = 0;

    // cases num => code number
    int max_nb_nodes = 8;
    std::vector<unsigned> case_nums( 1 << ( max_nb_nodes + 1 ), 0 ); // case => code number (generic case by default)
    for( int nb_nodes = 3; nb_nodes <= max_nb_nodes; ++nb_nodes ) {
        for( unsigned outside_case = 0; outside_case < ( 1 << nb_nodes ); ++outside_case ) {
            // make the code
            std::ostringstream os;
            make_case( os, nb_nodes, outside_case );

            // get number
            auto iter = code_map.find( os.str() );
            if ( iter == code_map.end() )
                iter = code_map.insert( iter, { os.str(), code_map.size() } );

            // store the case
            unsigned code_val = outside_case | ( 1 << nb_nodes );
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
        // the generic case is manually written
        if ( iter.second == 0 )
            continue;
        std::cout << "case_" << iter.second << ": {\n";
        std::cout << iter.first;
        std::cout << "    break;\n";
        std::cout << "}\n";
    }
    std::cout << "case_0:\n";
    std::cout << "    TODO; // generic case\n";
    std::cout << "    break;\n";
}
