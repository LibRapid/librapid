#ifndef NDARRAY_SCALAR_ARRAY_ARITHMETIC
#define NDARRAY_SCALAR_ARRAY_ARITHMETIC

#include <cstring> // For memset

namespace ndarray
{
	namespace arithmetic
	{
		template<typename A, typename B, typename C,
			typename E,
			typename S_b, typename S_c,
			typename LAMBDA>
			ND_INLINE void scalar_op_array(A *__restrict src_a, B *__restrict src_b, C *__restrict src_c,
										   const basic_extent<E> &extent,
										   const basic_stride<S_b> &stride_b,
										   const basic_stride<S_c> &stride_c,
										   LAMBDA op)
		{
			nd_int idim = 0;
			nd_int ndim = extent.ndim();

			const auto *__restrict _extent = extent.get_extent_alt();
			const auto *__restrict _stride_b = stride_b.get_stride_alt();
			const auto *__restrict _stride_c = stride_c.get_stride_alt();

			// All strides are non-trivial
			nd_int mode = 1;

			// All strides trivial
			if (stride_b.is_trivial() && stride_c.is_trivial())
				mode = 0;

			const auto end = math::product(extent.get_extent(), extent.ndim());

			nd_int coord[ND_MAX_DIMS]{};

			switch (mode)
			{
				case 0:
					{
						if (end > 100000)
						{
						#pragma omp parallel for shared(src_a, src_b, src_c, op, end) default(none) num_threads(ND_NUM_THREADS)
							for (long long i = 0; i < (long long) end; ++i)
								src_c[i] = op(*src_a, src_b[i]);
						}
						else
						{
							for (nd_int i = 0; i < end; ++i)
								src_c[i] = op(*src_a, src_b[i]);
						}
						break;
					}
				case 1:
					{
						do
						{
							*src_c = op(*src_a, *src_b);

							for (idim = 0; idim < ndim; ++idim)
							{
								if (++coord[idim] == _extent[idim])
								{
									coord[idim] = 0;
									src_b = src_b - (_extent[idim] - 1) * _stride_b[idim];
									src_c = src_c - (_extent[idim] - 1) * _stride_c[idim];
								}
								else
								{
									src_b = src_b + _stride_b[idim];
									src_c = src_c + _stride_c[idim];
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

#endif // NDARRAY_SCALAR_ARRAY_ARITHMETIC
