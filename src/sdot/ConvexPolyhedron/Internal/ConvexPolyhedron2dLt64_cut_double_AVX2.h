#ifdef __AVX2__
void ConvexPolyhedron2dLt64_cut( double *px, double *py, std::size_t *pi, int &nodes_size, const double *cut_x, const double *cut_y, const double *cut_s, const std::size_t *cut_i, int cn );
#endif // __AVX2__
