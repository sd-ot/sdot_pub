RUN = nsmake run -g3 # -D__AVX512F__ # --exec-with "valgrind --num-callers=64"

all:
	# ${RUN} tests/Support/test_Simd.cpp
	# ${RUN} tests/benchmarks/bench_Lgrid_2D_large.cpp
	# ${RUN} tests/benchmarks/bench_ConvexPolyhedron3b.cpp
	${RUN} tests/ConvexPolyhedron/test_ConvexPolyhedron3d.cpp

smurf:
	${RUN} tests/benchmarks/bench_Lgrid.cpp
	# ${RUN} tests/benchmarks/bench_ConvexPolyhedron2.cpp
	# ${RUN} tests/benchmarks/bench_ConvexPolyhedron3.cpp
	# ${RUN} tests/test_ConvexPolyhedron2.cpp
	# ${RUN} tests/benchmarks/bench_Zgrid_2D.cpp
	# ${RUN} tests/benchmarks/bench_Zgrid_2D.cpp
	# ${RUN} tests/benchmarks/bench_CGAL.cpp
	# ${RUN} tests/benchmarks/bench_CGAL_3D.cpp
	# ${RUN} tests/test_RadixSort.cpp
	# ${RUN} -I/usr/local/include/geogram1 -g3 tests/benchmarks/bench_geogram_delaunay_2D.cpp
	# ${RUN} -I/usr/local/include/geogram1 -g3 tests/benchmarks/bench_geogram_3D.cpp
	# ${RUN} tests/benchmarks/make_distrib_voro.cpp
	# ${RUN} tests/test_IntrusivePool.cpp
	# ${RUN} tests/test_PointerWithOffset.cpp
	# ${RUN} tests/test_ConvexPolyhedron3.cpp
	# ${RUN} tests/Grids/test_LGrid_2D.cpp
	# ${RUN} --exec-with valgrind tests/test_solve_2D.cpp
	# ${RUN} tests/benchmarks/distributions/make_distributions.cpp
	# python tests/benchmarks/distributions/make_distributions.py
	# ${RUN} tests/benchmarks/bench_geogram_3D.cpp . 
