#include <iostream>
#include <sstream>
#include <bitset>
#include <vector>
#include <map>

void make_case( std::ostream &os, unsigned nb_nodes, std::bitset<8> outside_nodes ) {
    bool all = true;
    for( unsigned i = 0; i < nb_nodes; ++i )
        all &= outside_nodes[ i ];

    // fully outside
    if ( all ) {
        os << "    // remove elem\n";
        os << "    faces_to_rem[ nb_faces_to_rem++ ] = num_face;\n";
        os << "    faces.node_masks[ num_face ] = 0;\n";
    }
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
