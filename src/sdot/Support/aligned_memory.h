#pragma once

#include <cstdlib>

namespace sdot {

void *aligned_malloc( std::size_t size, std::size_t alignment );
void  aligned_free  ( void *pointer );

}
