#include "IntrusivePool.h"

namespace sdot {

template<class T,int bs>
IntrusivePool<T,bs>::IntrusivePool() {
    last_bucket = nullptr;
    last_active = nullptr;
    last_free   = nullptr;
}

template<class T,int bs>
IntrusivePool<T,bs>::~IntrusivePool() {
    for( Bucket *b = last_bucket; b; ) {
        Bucket *p = b->prev;
        delete b;
        b = p;
    }
}

template<class T,int bs>
T *IntrusivePool<T,bs>::create() {
    if ( last_free == nullptr ) {
        Bucket *b = new Bucket;
        b->prev = last_bucket;
        last_bucket = b;

        for( int i = 0; i < nb_item_per_bucket; ++i ) {
            b->items[ i ].next_in_pool = last_free;
            last_free = b->items + i;
        }
    }

    // remove from the free list
    T *res = last_free;
    last_free = res->next_in_pool;

    // add to the active list
    res->prev_in_pool = last_active;
    res->next_in_pool = nullptr;
    if ( last_active )
        last_active->next_in_pool = res;
    last_active = res;

    return res;
}


template<class T, int bs>
void IntrusivePool<T,bs>::clear() {
    if ( last_bucket ) {
        while( Bucket *p = last_bucket->prev ) {
            delete last_bucket;
            last_bucket = p;
        }

        last_active = nullptr;
        last_free   = nullptr;

        for( int i = 0; i < nb_item_per_bucket; ++i ) {
            last_bucket->items[ i ].next_in_pool = last_free;
            last_free = last_bucket->items + i;
        }
    }
}

template<class T,int bs>
void IntrusivePool<T,bs>::free( T *item ) {
    // remove from the active list
    if ( item->next_in_pool )
        item->next_in_pool->prev_in_pool = item->prev_in_pool;
    else
        last_active = item->prev_in_pool;
    if ( item->prev_in_pool )
        item->prev_in_pool->next_in_pool = item->next_in_pool;

    // add to the free list
    item->next_in_pool = last_free;
    last_free = item;
}

template<class T,int bs>
void IntrusivePool<T,bs>::foreach( const std::function<void(T &)> &f ) const {
    for( T *v = last_active; v; v = v->prev_in_pool )
        f( *v );
}


} // namespace sdot

