#ifndef NDARRAY_ARRAY_ARITHMETIC
#define NDARRAY_ARRAY_ARITHMETIC

#include <librapid/math/rapid_math.hpp>
#include <cstring> // For memset

namespace librapid
{
	namespace ndarray
	{
		namespace arithmetic
		{
			template<typename A, typename B,
				typename E,
				typename S_a, typename S_b,
				typename LAMBDA>
				ND_INLINE void array_op(A *__restrict src_a, B *__restrict src_b,
										const basic_extent<E> &extent,
										const basic_stride<S_a> &stride_a,
										const basic_stride<S_b> &stride_b,
										LAMBDA op)
			{
				// Counters
				nd_int idim = 0;
				nd_int ndim = extent.ndim();

				// Create pointers here so repeated function calls
				// are not needed
				const auto *__restrict _extent = extent.get_extent();
				const auto *__restrict _stride_a = stride_a.get_stride();
				const auto *__restrict _stride_b = stride_b.get_stride();

				// All strides are non-trivial
				nd_int mode = 1;

				// All strides trivial
				if (stride_a.is_trivial() && stride_b.is_trivial())
					mode = 0;

				// Reduce calculations required
				const auto end = math::product(extent.get_extent(), extent.ndim());

				// The index in the array being calculated
				nd_int coord[ND_MAX_DIMS]{};

				switch (mode)
				{
					case 0:
						{
							// Only use OpenMP if size is above a certain threshold
							// to improve speed for small arrays (overhead of multiple
							// threads makes it slower for small arrays)
							if (end > 100000)
							{
								long long e = (long long) end;
							#pragma omp parallel for shared(src_a, src_b, op, e) default(none) num_threads(ND_NUM_THREADS)
								for (long long i = 0; i < e; ++i)
									src_a[i] = op(src_b[i]);
							}
							else
							{
								for (nd_int i = 0; i < end; ++i)
									src_a[i] = op(src_b[i]);
							}
							break;
						}
					case 1:
						{
							// Iterate over the array using it's stride and extent
							do
							{
								*src_a = op(*src_b);

								for (idim = 0; idim < ndim; ++idim)
								{
									if (++coord[idim] == _extent[idim])
									{
										coord[idim] = 0;
										src_a = src_a - (_extent[idim] - 1) * _stride_a[idim];
										src_b = src_b - (_extent[idim] - 1) * _stride_b[idim];
									}
									else
									{
										src_a = src_a + _stride_a[idim];
										src_b = src_b + _stride_b[idim];
										break;
									}
								}
							} while (idim < ndim);
							break;
						}
				}
			}
		}
	}
}

#endif // NDARRAY_ARRAY_ARITHMETIC