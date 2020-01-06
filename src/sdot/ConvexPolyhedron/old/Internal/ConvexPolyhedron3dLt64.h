#pragma once

#include "ConvexPolyhedron3dLt64_Face.h"
#include "ConvexPolyhedronOpt.h"
#include "../ConvexPolyhedron.h"
#include <vector>

namespace sdot {

/**
  Pc must contain
    - CI => cut info
    - TF (double, ...) => floating point type

*/
template<class Pc>
class alignas( 64 ) ConvexPolyhedron<Pc,3,ConvexPolyhedronOpt::Lt64> {
public:
    using                              TF                    = typename Pc::TF;    ///< floating point type
    using                              CI                    = typename Pc::CI;    ///< cut info
    using                              Pt                    = Point<TF,3>;      ///< point type

    using                              Lt64NodeBlock         = ConvexPolyhedron3Lt64NodeBlock<Pc>;
    using                              Lt64FaceBlock         = ConvexPolyhedron3Lt64FaceBlock<Pc>;
    using                              Bound                 = ConvexPolyhedron3Lt64_Face<Pc,ConvexPolyhedron<Pc,3,ConvexPolyhedronOpt::Lt64>>;
    using                              Node                  = Lt64NodeBlock;
    static constexpr int               dim                   = 3;

    struct                             Box                   { ConvexPolyhedron cp; };

    /**/                               ConvexPolyhedron      ( Pt p0, Pt p1, CI cut_id = {} );
    /**/                               ConvexPolyhedron      ();

    ConvexPolyhedron&                  operator=             ( const ConvexPolyhedron &that );
    ConvexPolyhedron&                  operator=             ( const Box &that );

    // information
    void                               write_to_stream       ( std::ostream &os ) const;
    std::size_t                        nb_nodes              () const;
    void                               check                 () const;

    bool                               empty                 () const;
    bool                               valid                 () const;

    Pt                                 node                  ( std::size_t index ) const;

    void                               for_each_bound        ( const std::function<void( const Bound &boundary_item )> &f ) const;
    void                               for_each_node         ( const std::function<void( Pt )> &f ) const;

    // geometric modifications
    template                           <int flags,class Fu>
    void                               plane_cut             ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts, N<flags>, const Fu &fu ); ///<
    void                               plane_cut             ( std::array<const TF *,dim> cut_dir, const TF *cut_ps, const CI *cut_id, std::size_t nb_cuts ); ///<

private:
    friend class                       ConvexPolyhedron3Lt64_Face<Pc,ConvexPolyhedron<Pc,3,ConvexPolyhedronOpt::Lt64>>;
    static constexpr int               max_nb_nodes          = 64;
    static constexpr int               max_nb_edges          = max_nb_nodes * max_nb_nodes;
    static constexpr int               max_nb_faces          = ConvexPolyhedron3Lt64FaceBlock<Pc>::max_nb_faces_per_cell;

    void                               update_cp_gen         ();

    // aligned structures
    ConvexPolyhedron3Lt64NodeBlock<Pc> nodes;                ///<
    ConvexPolyhedron3Lt64FaceBlock<Pc> faces;                ///<

    int                                nodes_size;
    int                                faces_size;

    std::vector<std::uint8_t>          additional_nums;      ///< used if at least 1 face has more than 16 nodes (meaning that we have to use another ConvexPolyhedron class)

    std::uint64_t                      edge_num_cut_procs[ max_nb_edges ]; ///< to be compared to this->num_cut_proc
    std::uint8_t                       edge_cuts         [ max_nb_edges ]; ///< num node for each possible edge

    std::uint64_t                      num_cut_proc;

    ConvexPolyhedron<Pc,3>             cp_gen;               ///< used if dat does not fit
};

} // namespace sdot

#include "ConvexPolyhedron3dLt64.tcc"
