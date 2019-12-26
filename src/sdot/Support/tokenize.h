#pragma once

#include <sstream>
#include <vector>
#include <string>

std::vector<std::string> tokenize( const std::string line, char sep = ' ' ) {
    std::vector<std::string> words;
    std::istringstream s( line );
    std::string word;
    while( std::getline( s, word, sep ) )
        if ( ! word.empty() )
        words.push_back( word );
    return words;
}
