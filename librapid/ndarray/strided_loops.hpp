#ifndef NDARRAY_STRIDED_LOOPS
#define NDARRAY_STRIDED_LOOPS

#define ND_BEGIN_INDEXED_LOOP_SINGLE(a_i, idim, ndim) \
auto *__restrict coord = new nd_int[ndim]; \
memset(coord, 0, sizeof(nd_int) * ndim); \
idim = 0; a_i = 0; \
do {
#define ND_INDEXED_LOOP_SINGLE(_extent, _stride_a, a_i, src_a, coord, idim, ndim) \
	for (idim = 0; idim < ndim; ++idim) \
	{ \
		if (++coord[idim] == _extent[idim]) \
		{ \
			coord[idim] = 0; \
			a_i = a_i - (_extent[idim] - 1) * _stride_a[idim]; \
		} \
		else \
		{ \
			a_i = a_i + _stride_a[idim]; \
			break; \
		} \
	} \
} while (idim < ndim); \
delete[] coord;

#define ND_BEGIN_INDEXED_LOOP_DOUBLE(a_i, b_i, idim, ndim) \
auto *__restrict coord = new nd_int[ndim]; \
memset(coord, 0, sizeof(nd_int) * ndim); \
idim = 0; a_i = 0; b_i = 0; \
do {
#define ND_INDEXED_LOOP_DOUBLE(_extent, _stride_a, _stride_b, a_i, b_i, src_a, src_b, coord, idim, ndim) \
	for (idim = 0; idim < ndim; ++idim) \
	{ \
		if (++coord[idim] == _extent[idim]) \
		{ \
			coord[idim] = 0; \
			a_i = a_i - (_extent[idim] - 1) * _stride_a[idim]; \
			b_i = b_i - (_extent[idim] - 1) * _stride_b[idim]; \
		} \
		else \
		{ \
			a_i = a_i + _stride_a[idim]; \
			b_i = b_i + _stride_b[idim]; \
			break; \
		} \
	} \
} while (idim < ndim); \
delete[] coord;

#define ND_BEGIN_INDEXED_LOOP_TRIPLE(a_i, b_i, c_i, idim, ndim) \
auto *__restrict coord = new nd_int[ndim]; \
memset(coord, 0, sizeof(nd_int) * ndim); \
idim = 0; a_i = 0; b_i = 0; c_i = 0; \
do {
#define ND_INDEXED_LOOP_TRIPLE(_extent, _stride_a, _stride_b, _stride_c, a_i, b_i, c_i, src_a, src_b, src_c, coord, idim, ndim) \
	for (idim = 0; idim < ndim; ++idim) \
	{ \
		if (++coord[idim] == _extent[idim]) \
		{ \
			coord[idim] = 0; \
			a_i = a_i - (_extent[idim] - 1) * _stride_a[idim]; \
			b_i = b_i - (_extent[idim] - 1) * _stride_b[idim]; \
			c_i = c_i - (_extent[idim] - 1) * _stride_c[idim]; \
		} \
		else \
		{ \
			a_i = a_i + _stride_a[idim]; \
			b_i = b_i + _stride_b[idim]; \
			c_i = c_i + _stride_c[idim]; \
			break; \
		} \
	} \
} while (idim < ndim); \
delete[] coord;

#endif // NDARRAY_STRIDED_LOOPS