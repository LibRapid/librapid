#ifndef LIBRAPID_MUTLIARRAY_OPERATIONS
#define LIBRAPID_MUTLIARRAY_OPERATIONS

#include <librapid/config.hpp>
#include <librapid/autocast/autocast.hpp>

namespace librapid
{
	namespace imp
	{
		/**
		 * \rst
		 *
		 * Provided with some data, a source location and a target location, set a
		 * result memory pointer to a (potentially new) memory address on the target
		 * accelerator.
		 *
		 * If the ``src`` and ``target`` locations are equal, then the ``data``
		 * pointer is simply copied to the ``result`` value. Otherwise memory will
		 * be allocated and the data copied to the required location.
		 *
		 * The return value is the required memory freeing mode for the new memory
		 * address
		 *
		 * - 0 = do not free memory
		 * - 1 = free memory on host (normal RAM)
		 * - 2 = free memory on device (GPU memory)
		 *
		 * Parameters
		 * ----------
		 * src: Accelerator
		 *		The accelerator on which the source data is located
		 * target: Accelerator
		 *		The target location for the data
		 * data: ``A *``
		 *		The source data
		 * result: ``A **``
		 *		Where the result data will be placed
		 * size: integer
		 *		The number of *elements* in ``data``
		 *
		 * \endrst
		 */
		template<typename A>
		int makeSameAccelerator(Accelerator src, Accelerator target,
								A *__restrict data, A **__restrict result,
								size_t size)
		{
			// Freeing information
			// 0 = no free
			// 1 = free()
			// 2 = cudaFree()

			auto freeMode = -1;
			if (src == target)
			{
				freeMode = 0;
				*result = data;
			}
			else
			{
				if (target == Accelerator::CPU)
				{
					// Copy A from GPU to CPU
				#ifdef LIBRAPID_HAS_CUDA
					// Allocate memory
					freeMode = 1;
					*result = (A *) malloc(sizeof(A) * size);

					// Copy data
				#ifdef LIBRAPID_CUDA_STREAM
					cudaSafeCall(cudaMemcpyAsync(*result, data, sizeof(A) * size,
								 cudaMemcpyDeviceToHost, cudaStream));
				#else
					cudaSafeCall(cudaDeviceSynchronize());
					cudaSafeCall(cudaMemcpy(*result, data, sizeof(A) * size,
								 cudaMemcpyDeviceToHost));
				#endif // LIBRAPID_CUDA_STREAM
				#else
					throw std::invalid_argument("GPU support was not enabled, so"
												" calculations involving the GPU"
												" are not possible");
				#endif // LIBRAPID_HAS_CUDA
				}
				else if (target == Accelerator::GPU)
				{
					// Copy A from CPU to GPU

				#ifdef LIBRAPID_HAS_CUDA
					freeMode = 2;
					// *resultA = (A *) malloc(sizeof(A) * size);

					// Copy data
				#ifdef LIBRAPID_CUDA_STREAM
					// Allocate memory
					cudaSafeCall(cudaMallocAsync(result, sizeof(A) * size,
								 cudaStream));

					cudaSafeCall(cudaMemcpyAsync(*result, data, sizeof(A) * size,
								 cudaMemcpyHostToDevice, cudaStream));
				#else
					cudaSafeCall(cudaMalloc(result, sizeof(A) * size));

					cudaSafeCall(cudaDeviceSynchronize());
					cudaSafeCall(cudaMemcpy(*result, data, sizeof(A) * size,
								 cudaMemcpyHostToDevice));
				#endif // LIBRAPID_CUDA_STREAM
				#else
					throw std::invalid_argument("GPU support was not enabled, so"
												" calculations involving the GPU"
												" are not possible");
				#endif // LIBRAPID_HAS_CUDA
				}
			}

			return freeMode;
		}

		/**
		 * \rst
		 *
		 * Free memory with a given freeing mode
		 *
		 * - 0 = do not free memory
		 * - 1 = free host memory (``free()``)
		 * - 2 = free device memory (``cudaFree()``)
		 *
		 * \endrst
		 */
		template<typename A>
		void freeWithMode(A *__restrict data, int mode)
		{
			// Freeing information
			// 0 = no free
			// 1 = free()
			// 2 = cudaFree()
			if (mode == 1)
			{
				free(data);
			}
			else if (mode == 2)
			{
			#ifdef LIBRAPID_HAS_CUDA
			#ifdef LIBRAPID_CUDA_STREAM
				cudaSafeCall(cudaFreeAsync(data, cudaStream));
			#else
				cudaSafeCall(cudaDeviceSynchronize());
				cudaSafeCall(cudaFree(data));
			#endif // LIBRAPID_CUDA_STREAM
			#endif // LIBRAPID_HAS_CUDA
			}
			else if (mode != 0)
			{
				throw std::invalid_argument("Invalid free mode for binary "
											"operation");
			}
		}

		/**
		 * \rst
		 *
		 * Perform a unary operation on trivially strided, contiguous memory.
		 *
		 * An operation struct can be passed to the ``op`` parameter of this
		 * function. It must contain a ``func``, being a lambda taking a single
		 * templated value, and returning a single value of the same type.
		 *
		 * If CUDA support is enabled, the operation struct must also contain a
		 * small kernel operation stored as a string. An example kernel is
		 * ``"b = 2 * a"`` -- "b" and "a" are scalar values corresponding to the
		 * provided input data. This example kernel will multiply every value in
		 * ``a`` by two, and will store the result in ``b``.
		 *
		 *
		 * Parameters
		 * ----------
		 * locnA: Accelerator
		 *		The location of ``a``
		 * locnB: Accelerator
		 *		The location of ``b``
		 * a: ``A *``
		 *		Trivial array
		 * b: ``B *``
		 *		Trivial array
		 * size: integer
		 *		The number of *elements* in ``a`` (and therefore ``b``)
		 * op: Operation struct (containing "name", "kernel" and "func")
		 *
		 * \endrst
		 */
		template<typename A, typename B, class FUNC>
		void multiarrayUnaryOpTrivial(Accelerator locnA, Accelerator locnB,
									  A *__restrict a, B *__restrict b,
									  size_t size, const FUNC &op)
		{
			if (locnA != locnB)
			{
				// Copy A to be on the same accelerator as B
				A *tempA = nullptr;
				int freeMode = makeSameAccelerator(locnA, locnB, a, &tempA, size);
				multiarrayUnaryOpTrivial(locnB, locnB, tempA, b, size, op);
				freeWithMode(tempA, freeMode);
			}
			else
			{
				if (locnA == Accelerator::CPU)
				{
					if (size < 10000)
					{
						for (size_t i = 0; i < size; ++i)
							b[i] = (B) op(a[i]);
					}
					else
					{
					#pragma omp parallel for shared(a, b, size, op) num_threads(LIBRAPID_ARITHMETIC_THREADS) default(none)
						for (lr_int i = 0; i < (lr_int) size; ++i)
							b[i] = (B) op(a[i]);
					}
				}
			#ifdef LIBRAPID_HAS_CUDA
				else
				{
					using jitify::reflection::Type;

					std::string kernel = "unaryKernel\n";
					kernel += "__constant__ int LIBRAPID_MAX_DIMS = "
						+ std::to_string(LIBRAPID_MAX_DIMS) + ";";
					kernel += R"V0G0N(
					template<typename A, typename B, typename C>
					__global__
					void unaryFunc(const A *__restrict arrayA,
								   const B *__restrict arrayB,
								   C *__restrict arrayC, size_t size)
					{
						unsigned int kernelIndex = blockDim.x * blockIdx.x
												   + threadIdx.x;

						if (kernelIndex < size) {
							const auto &a = arrayA[kernelIndex];
							auto &b = arrayB[kernelIndex];
					)V0G0N";

					kernel += op.kernel;
					kernel += "\n}\n}";

					static jitify::JitCache kernelCache;
					jitify::Program program = kernelCache.program(kernel, 0);

					unsigned int threadsPerBlock, blocksPerGrid;

					// Use 1 to 256 threads per block
					if (size < 256)
					{
						threadsPerBlock = (unsigned int) size;
						blocksPerGrid = 1;
					}
					else
					{
						threadsPerBlock = 256;
						blocksPerGrid = ceil(double(size) / double(threadsPerBlock));
					}

					dim3 grid(blocksPerGrid);
					dim3 block(threadsPerBlock);

				#ifdef LIBRAPID_CUDA_STREAM
					jitifyCall(program.kernel("unaryFunc")
							   .instantiate(Type<A>(), Type<B>())
							   .configure(grid, block, 0, cudaStream)
							   .launch(a, b, size));
				#else
					jitifyCall(program.kernel("unaryFunc")
							   .instantiate(Type<A>(), Type<B>())
							   .configure(grid, block)
							   .launch(a, b, size));
				#endif // LIBRAPID_CUDA_STREAM
				}
			}
		#endif // LIBRAPID_HAS_CUDA
		}

		/**
		 * \rst
		 *
		 * Perform a binary operation on trivially strided, contiguous memory.
		 *
		 * An operation struct can be passed to the ``op`` parameter of this
		 * function. It must contain a ``func``, being a lambda taking two
		 * templated values which returns a single ``auto`` value.
		 *
		 * If CUDA support is enabled, the operation struct must also contain a
		 * small kernel operation stored as a string. An example kernel is
		 * ``"c = a + b"`` -- "c", "b" and "a" are scalar values corresponding to the
		 * provided input data. This example kernel will result in the the data being
		 * added together, and the result being stored in ``c``
		 *
		 * If using scalar values as inputs, set the corresponding ``?IsScalar``
		 * parameter to true. Only one scalar input is allowed.
		 *
		 * Parameters
		 * ----------
		 * locnA: Accelerator
		 *		The location of ``a``
		 * locnB: Accelerator
		 *		The location of ``b``
		 * locnC: Accelerator
		 *		The location of the result data ``c``
		 * a: ``A *``
		 *		Trivial array
		 * b: ``B *``
		 *		Trivial array
		 * c: ``C *``
		 *		Trivial array
		 * aIsScalar: Boolean
		 *		If true, the input data ``a`` will be treated as a scalar
		 * bIsScalar: Boolean
		 *		If true, the input data ``b`` will be treated as a scalar
		 * size: integer
		 *		The number of *elements* in ``a`` (and therefore ``b``)
		 * op: Operation struct (containing "name", "kernel" and "func")
		 *
		 * \endrst
		 */
		template<typename A, typename B, typename C, class FUNC>
		void multiarrayBinaryOpTrivial(Accelerator locnA, Accelerator locnB,
									   Accelerator locnC, A *__restrict a,
									   B *__restrict b, C *__restrict c,
									   bool aIsScalar, bool bIsScalar,
									   size_t size, const FUNC &op)
		{
			if (aIsScalar && bIsScalar)
				throw std::invalid_argument("Only a single scalar input is allowed "
											" in a trivial binary operation, but "
											"two were specified");

			if (!(locnA == locnB && locnB == locnC))
			{
				// Locations are different, so make A and B have the same
				// accelerator as the result array (C)

				A *tempA = nullptr;
				B *tempB = nullptr;

				int freeA = makeSameAccelerator(locnA, locnC, a, &tempA,
												aIsScalar ? 1 : size);

				int freeB = makeSameAccelerator(locnB, locnC, b, &tempB,
												bIsScalar ? 1 : size);

				// Run the function again
				multiarrayBinaryOpTrivial(locnC, locnC, locnC, tempA, tempB, c,
										  aIsScalar, bIsScalar, size, op);

				// Free the data
				freeWithMode(tempA, freeA);
				freeWithMode(tempB, freeB);
			}
			else
			{
				if (locnA == Accelerator::CPU)
				{
					if (aIsScalar)
					{
						// Use *a rather than a[i]
						if (size < 2500)
						{
							for (size_t i = 0; i < size; ++i)
								c[i] = (C) op(*a, b[i]);
						}
						else
						{
						#pragma omp parallel for shared(a, b, c, size, op) num_threads(LIBRAPID_ARITHMETIC_THREADS) default(none)
							for (lr_int i = 0; i < (lr_int) size; ++i)
								c[i] = (C) op(*a, b[i]);
						}
					}
					else if (bIsScalar)
					{
						// Use *b rather than b[i]
						if (size < 2500)
						{
							for (size_t i = 0; i < size; ++i)
								c[i] = (C) op(a[i], *b);
						}
						else
						{
						#pragma omp parallel for shared(a, b, c, size, op) num_threads(LIBRAPID_ARITHMETIC_THREADS) default(none)
							for (lr_int i = 0; i < (lr_int) size; ++i)
								c[i] = (C) op(a[i], *b);
						}
					}
					else
					{
						// Use a[i] and b[i]
						if (size < 2500)
						{
							for (size_t i = 0; i < size; ++i)
								c[i] = (C) op(a[i], b[i]);
						}
						else
						{
						#pragma omp parallel for shared(a, b, c, size, op) num_threads(LIBRAPID_ARITHMETIC_THREADS) default(none)
							for (lr_int i = 0; i < (lr_int) size; ++i)
								c[i] = (C) op(a[i], b[i]);
						}
					}
				}
			#ifdef LIBRAPID_HAS_CUDA
				else
				{
					using jitify::reflection::Type;

					std::string kernel = "binaryKernel\n";
					kernel += "__constant__ int LIBRAPID_MAX_DIMS = "
						+ std::to_string(LIBRAPID_MAX_DIMS) + ";";
					kernel += R"V0G0N(
					template<typename A, typename B, typename C>
					__global__
					void binaryFunc(const A *__restrict arrayA,
									const B *__restrict arrayB,
									C *__restrict arrayC, size_t size)
					{
						unsigned int kernelIndex = blockDim.x * blockIdx.x
												   + threadIdx.x;

						if (kernelIndex < size) {
							auto &c = arrayC[kernelIndex];
					)V0G0N";

					if (aIsScalar)
					{
						kernel += "const auto &a = *arrayA;\n"
							"const auto &b = arrayB[kernelIndex];";
					}
					else if (bIsScalar)
					{
						kernel += "const auto &a = arrayA[kernelIndex];\n"
							"const auto &b = *arrayB;";
					}
					else
					{
						kernel += "const auto &a = arrayA[kernelIndex];\n"
							"const auto &b = arrayB[kernelIndex];";
					}

					kernel += op.kernel;
					kernel += "\n}\n}";

					// const std::vector<std::string> params = {
					// };

					static jitify::JitCache kernelCache;
					jitify::Program program = kernelCache.program(kernel, 0);

					unsigned int threadsPerBlock, blocksPerGrid;

					// Use 1 to 256 threads per block
					if (size < 256)
					{
						threadsPerBlock = (unsigned int) size;
						blocksPerGrid = 1;
					}
					else
					{
						threadsPerBlock = 256;
						blocksPerGrid = ceil(double(size) / double(threadsPerBlock));
					}

					dim3 grid(blocksPerGrid);
					dim3 block(threadsPerBlock);

				#ifdef LIBRAPID_CUDA_STREAM
					jitifyCall(program.kernel("binaryFunc")
							   .instantiate(Type<A>(), Type<B>(), Type<C>())
							   .configure(grid, block, 0, cudaStream)
							   .launch(a, b, c, size));
				#else
					jitifyCall(program.kernel("binaryFunc")
							   .instantiate(Type<A>(), Type<B>(), Type<C>())
							   .configure(grid, block)
							   .launch(a, b, c, size));
				#endif // LIBRAPID_CUDA_STREAM
				}
			#endif // LIBRAPID_HAS_CUDA
			}
		}
	}
}

#endif // LIBRAPID_MUTLIARRAY_OPERATIONS