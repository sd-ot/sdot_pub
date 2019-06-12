#include "../Support/aligned_memory.h"
#include "../Support/TODO.h"
#include "../Support/P.h"
#include "ConvexPolyhedron2.h"
#include <immintrin.h>
#include <iomanip>

namespace sdot {

template<class Pc>
ConvexPolyhedron2<Pc>::ConvexPolyhedron2( const Box &box, CI cut_id ) : ConvexPolyhedron2() {
    resize( 4 );

    // points
    Node &n0 = node( 0 );
    Node &n1 = node( 1 );
    Node &n2 = node( 2 );
    Node &n3 = node( 3 );

    n0.x = box.p0[ 0 ]; n0.y = box.p0[ 1 ];
    n1.x = box.p1[ 0 ]; n1.y = box.p0[ 1 ];
    n2.x = box.p1[ 0 ]; n2.y = box.p1[ 1 ];
    n3.x = box.p0[ 0 ]; n3.y = box.p1[ 1 ];

    // normals
    n0.dir_x =  0; n0.dir_y = -1;
    n1.dir_x = +1; n1.dir_y =  0;
    n2.dir_x =  0; n2.dir_y = +1;
    n3.dir_x = -1; n3.dir_y =  0;

    // cut_ids
    n0.cut_id.set( cut_id );
    n1.cut_id.set( cut_id );
    n2.cut_id.set( cut_id );
    n3.cut_id.set( cut_id );
}

template<class Pc>
ConvexPolyhedron2<Pc>::ConvexPolyhedron2() {
    size = 0;
    rese = block_size;
    nodes = new ( aligned_malloc( rese / block_size * sizeof( Node ), 32 ) ) Node;
}

template<class Pc>
ConvexPolyhedron2<Pc>::~ConvexPolyhedron2() {
    if ( rese )
        aligned_free( nodes );
}

template<class Pc>
void ConvexPolyhedron2<Pc>::write_to_stream( std::ostream &os ) const {
    //    for( TI i = 0; i < nb_nodes(); ++i ) {
    //        os << node( );
    //    }
    os << "pouet";
    //    os << std::setprecision( 17 );
    //    os << "cuts: [";
    //    for( TI i = 0; i < _nb_points; ++i )
    //        os << ( i ? "," : "" ) << "(" << point( i ) << ")";
    //    os << "]";
    //    if ( store_the_normals ) {
    //        os << " nrms: [";
    //        for( TI i = 0; i < _nb_points; ++i )
    //            os << ( i ? "," : "" ) << "(" << normal( i ) << ")";
    //        os << "]";
    //    }
    //    os << " sphere center: " << sphere_center << " sphere radius: " << sphere_radius;
}

template<class Pc> template<class F>
void ConvexPolyhedron2<Pc>::for_each_edge( const F &/*f*/ ) {
    TODO;
}

template<class Pc> template<class F>
void ConvexPolyhedron2<Pc>::for_each_node( const F &f ) {
    static_assert ( sizeof( Node ) % sizeof( TF ) == 0, "" );

    Node *ptr = nodes;
    for( TI i = 0; ; ++i ) {
        if ( i + block_size >= size ) {
            for( TI j = 0; j < size - i; ++j )
                f( ptr->local_at( j ) );
            break;
        }

        for( TI j = 0; j < block_size; ++j )
            f( ptr->local_at( j ) );
        i += block_size;
        ++ptr;
    }
}

template<class Pc>
typename ConvexPolyhedron2<Pc>::TI ConvexPolyhedron2<Pc>::nb_nodes() const {
    return size;
}

template<class Pc>
const typename ConvexPolyhedron2<Pc>::Node &ConvexPolyhedron2<Pc>::node( TI index ) const {
    return nodes->global_at( index );
}

template<class Pc>
typename ConvexPolyhedron2<Pc>::Node &ConvexPolyhedron2<Pc>::node( TI index ) {
    return nodes->global_at( index );
}

template<class Pc>
void ConvexPolyhedron2<Pc>::resize( TI new_size ) {
    if ( new_size > rese ) {
        TI old_nb_blocks = rese / block_size;
        Node *old_nodes = nodes;

        TI nb_blocks = ( new_size + block_size - 1 ) / block_size;
        rese = nb_blocks * block_size;

        nodes = reinterpret_cast<Node *>( aligned_malloc( nb_blocks * sizeof( Node ), 32 ) );
        for( TI i = 0; i < old_nb_blocks; ++i )
            new ( nodes + i ) Node( std::move( old_nodes[ i ] ) );

        if ( old_nb_blocks )
            aligned_free( old_nodes );
    }
    size = new_size;
}

template<class Pc>
bool ConvexPolyhedron2<Pc>::plane_cut( Pt origin, Pt normal, CI cut_id ) {
    return plane_cut( origin, normal, cut_id, N<1>() );
}

template<class Pc> template<int no>
bool ConvexPolyhedron2<Pc>::plane_cut( Pt /*origin*/, Pt /*normal*/, CI /*cut_id*/, N<no> /*normal_is_normalized*/ ) {
    return true;
}

} // namespace sdot
