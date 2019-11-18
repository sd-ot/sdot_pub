#ifndef SDOT_INTRUSIVE_POOL_H
#define SDOT_INTRUSIVE_POOL_H

#include <functional>

namespace sdot {

/**
  bs = size of a bucket in bytes

  Beware: T is assumed to have a trivial destructor.

  T must contain:
  - prev_in_pool
  - next_in_pool
*/
template<class T,int bs>
class IntrusivePool {
public:
    /**/    IntrusivePool      ();
    /**/   ~IntrusivePool      ();

    T      *create             ();
    void    clear              ();
    void    free               ( T *item );

    void    foreach            ( const std::function<void(T &)> &f ) const;
    bool    empty              () const { return last_active == nullptr; }

private:
    enum {  nb_item_per_bucket = ( bs - sizeof( void * ) ) / sizeof( T ) };
    struct  Bucket             { Bucket *prev; T items[ nb_item_per_bucket ]; };

    Bucket *last_bucket;       ///<
    T      *last_active;
    T      *last_free;
};

} // namespace sdot

#include "IntrusivePool.tcc"

#endif // SDOT_INTRUSIVE_POOL_H
