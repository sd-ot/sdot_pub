#ifndef SDOT_MemoryPool_H
#define SDOT_MemoryPool_H

#include <cstdint>
#include <utility>

/**
  Simple memory pool. Objects are not freed.
*/
class MemoryPool {
public:
    /* */    MemoryPool();
    /* */   ~MemoryPool();

    char*    allocate       ( std::size_t size, std::size_t alig );

    template                <class T,class... Args>
    T*       create         ( Args &&...args );

private:
    struct   Frame          { Frame *prev_frame; char content[ 8 ]; };
    union    Ptr            { char *cp; std::size_t vp; };

    Ptr      current_ptr;
    char*    ending_ptr;
    Frame*   last_frame;
};

#include "MemoryPool.tcc"

#endif // SDOT_MemoryPool_H
