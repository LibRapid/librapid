#ifndef LIBRAPID_MUTLIARRAY_OPERATIONS
#define LIBRAPID_MUTLIARRAY_OPERATIONS

#include <librapid/config.hpp>
#include <librapid/autocast/autocast.hpp>

namespace librapid::imp {
	#ifdef LIBRAPID_HAS_CUDA
	inline const jitify::detail::vector<std::string> cudaHeaders = {// CUDA_INCLUDE_DIRS,
			CUDA_INCLUDE_DIRS + std::string("/curand.h"),
			CUDA_INCLUDE_DIRS + std::string("/curand_kernel.h"),
			CUDA_INCLUDE_DIRS + std::string("/cublas_v2.h"),
			CUDA_INCLUDE_DIRS + std::string("/cublas_api.h"),
			CUDA_INCLUDE_DIRS + std::string("/cuda_fp16.h"),
			CUDA_INCLUDE_DIRS + std::string("/cuda_bf16.h")
	};

	inline const std::vector<std::string> cudaParams = {
			"--disable-warnings",
			"-std=c++17",
			std::string("-I") + CUDA_INCLUDE_DIRS
	};

	inline constexpr char complexHpp[] = R"V0G0N(
#ifndef LIBRAPID_CUSTOM_COMPLEX
#define LIBRAPID_CUSTOM_COMPLEX

namespace librapid
{
	template <class T>
	class Complex
	{
	public:
		Complex(const T &real_val = T(), const T &imag_val = T())
			: m_real(real_val), m_imag(imag_val)
		{}

		Complex &operator=(const T &val)
		{
			m_real = val;
			m_imag = 0;
			return *this;
		}

		template <class V>
		Complex(const Complex<V> &other)
			: Complex(static_cast<T>(other.real()), static_cast<T>(other.imag()))
		{}

		template <class V>
		Complex &operator=(const Complex<V> &other)
		{
			m_real = static_cast<T>(other.real());
			m_imag = static_cast<T>(other.imag());
			return *this;
		}

		Complex copy() const
		{
			return Complex<T>(m_real, m_imag);
		}

		Complex operator-() const
		{
			return Complex<T>(-m_real, -m_imag);
		}

		template<typename V>
		Complex operator+(const V &other) const
		{
			return Complex<T>(m_real + other, m_imag);
		}

		template<typename V>
		Complex operator-(const V &other) const
		{
			return Complex<T>(m_real - other, m_imag);
		}

		template<typename V>
		Complex operator*(const V &other) const
		{
			return Complex<T>(m_real * other, m_imag * other);
		}

		template<typename V>
		Complex operator/(const V &other) const
		{
			return Complex<T>(m_real / other, m_imag / other);
		}

		template<typename V>
		Complex &operator+=(const V &other)
		{
			m_real += other;
			return *this;
		}

		template<typename V>
		Complex &operator-=(const V &other)
		{
			m_real -= other;
			return *this;
		}

		template<typename V>
		Complex &operator*=(const V &other)
		{
			m_real *= other;
			m_imag *= other;
			return *this;
		}

		template<typename V>
		Complex &operator/=(const V &other)
		{
			m_real /= other;
			m_imag /= other;
			return *this;
		}

		template<typename V>
		Complex operator+(const Complex<V> &other) const
		{
			return Complex(m_real + other.real(),
						   m_imag + other.imag());
		}

		template<typename V>
		Complex operator-(const Complex<V> &other) const
		{
			return Complex(m_real - other.real(),
						   m_imag - other.imag());
		}

		template<typename V>
		Complex operator*(const Complex<V> &other) const
		{
			return Complex((m_real * other.real()) - (m_imag * other.imag()),
						   (m_real * other.imag()) + (m_imag * other.real()));
		}

		template<typename V>
		Complex operator/(const Complex<V> &other) const
		{
			return Complex((m_real * other.real()) + (m_imag * other.imag()) /
						   ((other.real() * other.real()) + (other.imag() * other.imag())),
						   (m_real * other.real()) - (m_imag * other.imag()) /
						   ((other.real() * other.real()) + (other.imag() * other.imag())));
		}

		template<typename V>
		Complex &operator+=(const Complex<V> &other)
		{
			m_real = m_real + other.real();
			m_imag = m_imag + other.imag();
			return *this;
		}

		template<typename V>
		Complex &operator-=(const Complex<V> &other)
		{
			m_real = m_real - other.real();
			m_imag = m_imag - other.imag();
			return *this;
		}

		template<typename V>
		Complex &operator*=(const Complex<V> &other)
		{
			m_real = (m_real * other.real()) - (m_imag * other.imag());
			m_imag = (m_real * other.imag()) + (imag() * other.real());
			return *this;
		}

		template<typename V>
		Complex &operator/=(const Complex<V> &other)
		{
			m_real = (m_real * other.real()) + (m_imag * other.imag()) /
				((other.real() * other.real()) + (other.imag() * other.imag()));
			m_imag = (m_real * other.real()) - (m_imag * other.imag()) /
				((other.real() * other.real()) + (other.imag() * other.imag()));
			return *this;
		}

		template<typename V>
		bool operator==(const Complex<V> &other) const
		{
			return m_real == other.real() && m_imag == other.imag();
		}

		template<typename V>
		bool operator!=(const Complex<V> &other) const
		{
			return !(*this == other);
		}

		T mag() const
		{
			return std::sqrt(m_real * m_real + m_imag * m_imag);
		}

		T angle() const
		{
			return std::atan2(m_real, m_imag);
		}

		Complex<T> log() const
		{
			return Complex<T>(std::log(mag()), angle());
		}

		Complex<T> conjugate() const
		{
			return Complex<T>(m_real, -m_imag);
		}

		Complex<T> reciprocal() const
		{
			return Complex<T>((m_real) / (m_real * m_real + m_imag * m_imag),
							  -(m_imag) / (m_real * m_real + m_imag * m_imag));
		}

		const T &real() const
		{
			return m_real;
		}

		T &real()
		{
			return m_real;
		}

		const T &imag() const
		{
			return m_imag;
		}

		T &imag()
		{
			return m_imag;
		}

		template<typename V>
		operator V() const
		{
			return m_real;
		}

	private:
		T m_real = 0;
		T m_imag = 0;
	};

	template<typename A, typename B>
		Complex<B> operator+(const A &a, const Complex<B> &b)
	{
		return Complex<B>(a) + b;
	}

	template<typename A, typename B>
		Complex<B> operator-(const A &a, const Complex<B> &b)
	{
		return Complex<B>(a) - b;
	}

	template<typename A, typename B>
		Complex<B> operator*(const A &a, const Complex<B> &b)
	{
		return Complex<B>(a) * b;
	}

	template<typename A, typename B>
		Complex<B> operator/(const A &a, const Complex<B> &b)
	{
		return Complex<B>(a) / b;
	}

	template<typename A, typename B>
		A &operator+=(A &a, const Complex<B> &b)
	{
		a += b.real();
		return a;
	}

	template<typename A, typename B>
		A &operator-=(A &a, const Complex<B> &b)
	{
		a -= b.real();
		return a;
	}

	template<typename A, typename B>
		A &operator*=(A &a, const Complex<B> &b)
	{
		a *= b.real();
		return a;
	}

	template<typename A, typename B>
		A &operator/=(A &a, const Complex<B> &b)
	{
		a /= b.real();
		return a;
	}
}

#endif
		)V0G0N";
	#endif // LIBRAPID_HAS_CUDA

	inline int makeSameAccelerator(RawArray &dst,
								   const RawArray &src,
								   int64_t size) {
		// Freeing information
		// 0 = no free
		// 1 = cudaFree()
		// 2 = free()

		auto freeMode = -1;
		if (dst.location == src.location) {
			freeMode = 0;
			dst = src;
		} else {
			if (src.location == Accelerator::CPU) {
				// Copy from CPU to GPU
				// Allocate memory

				freeMode = 1;
				rawArrayMalloc(dst, size);
				rawArrayMemcpy(dst, src, size);
			}
#ifdef LIBRAPID_HAS_CUDA
			else if (src.location == Accelerator::GPU) {
				// Copy A from GPU to CPU

				freeMode = 2;
				rawArrayMalloc(dst, size);
				rawArrayMemcpy(dst, src, size);
			}
#else
			else {
				throw std::invalid_argument("GPU support was not enabled, so"
											" calculations involving the GPU"
											" are not possible");
			}
#endif // LIBRAPID_HAS_CUDA
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
	inline void freeWithMode(RawArray &raw, int mode) {
		// Freeing information
		// 0 = no free
		// 1 = cudaFree()
		// 2 = free()
		if (mode == 0)
			return;

		if (mode == 1 || mode == 2)
			rawArrayFree(raw);
		else
			throw std::invalid_argument("Invalid free mode for binary operation");
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
	template<typename FUNC>
	inline void multiarrayUnaryOpTrivial(RawArray dst, const RawArray &src,
										 int64_t elems, const FUNC &op) {
		if (dst.location != src.location) {
			// Copy A to be on the same accelerator as B
			RawArray tempSrc = {(int64_t *) nullptr, dst.dtype, dst.location};
			rawArrayMalloc(tempSrc, elems);
			int freeMode = makeSameAccelerator(tempSrc, src, elems);
			multiarrayUnaryOpTrivial(dst, tempSrc, elems, op);
			freeWithMode(tempSrc, freeMode);
		} else {
			if (dst.location == Accelerator::CPU) {
				std::visit([&](auto *dstData, auto *srcData) {
					auto tempElems = elems;
					auto tempOp = op;

					using A = typename std::remove_pointer<decltype(dstData)>::type;
					using B = typename std::remove_pointer<decltype(srcData)>::type;

					if (elems < THREAD_THREASHOLD) {
						for (int64_t i = 0; i < tempElems; ++i)
							dstData[i] = tempOp(srcData[i], i);
					} else {
#pragma omp parallel for shared(dstData, srcData, tempElems, tempOp)
						for (int64_t i = 0; i < tempElems; ++i) {
							dstData[i] = tempOp(srcData[i], i);
						}
					}
				}, dst.data, src.data);
			}
#ifdef LIBRAPID_HAS_CUDA
			else {
				using jitify::reflection::Type;

				static double randSeed = seconds() * 10;

				std::string kernel = "unaryKernelTrivial\n";
				kernel += "__constant__ int LIBRAPID_MAX_DIMS = "
						  + std::to_string(LIBRAPID_MAX_DIMS) + ";\n";
				kernel += "#include <stdint.h>\n"
						  "#include <type_traits>\n"
						  "#include <" CUDA_INCLUDE_DIRS "/curand_kernel.h>\n"
						  "#include <" CUDA_INCLUDE_DIRS "/curand.h>\n\n";

				kernel += complexHpp;

				kernel += R"V0G0N(
					template<typename T_DST, typename A>
					__device__
					inline auto )V0G0N";

				kernel += op.name;
				kernel += R"V0G0N((const A &a, int64_t indexA, curandState_t *_curandState)
					{
					)V0G0N";
				kernel += op.kernel;

				kernel += R"V0G0N(}

					template<typename T_DST, typename T_SRC>
					__global__
					void unaryFuncTrivial(T_DST *__restrict dstData,
										  const T_SRC *__restrict srcData,
										  const uint64_t size,
										  curandState_t *_curandStates = nullptr)
					{
						int64_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;

						if (kernelIndex < size) {
					)V0G0N";
				kernel += "dstData[kernelIndex] = " + op.name +
						  "<T_DST, T_SRC>(srcData[kernelIndex], kernelIndex, _curandStates + kernelIndex);";
				kernel += "\n}\n}";

				static jitify::JitCache kernelCache;
				jitify::Program program = kernelCache.program(kernel, cudaHeaders, cudaParams);

				unsigned int threadsPerBlock, blocksPerGrid;

				// Use 1 to 512 threads per block
				if (elems < 512) {
					threadsPerBlock = (unsigned int) elems;
					blocksPerGrid = 1;
				} else {
					threadsPerBlock = 512;
					blocksPerGrid = ceil(double(elems) / double(threadsPerBlock));
				}

				dim3 grid(blocksPerGrid);
				dim3 block(threadsPerBlock);

				static curandState_t *curandGenerators = nullptr;
				static int64_t numGenerators = 0;
				if ((op.name == "fillRandom" || op.name == "fillRandomComplex") &&
					(numGenerators < threadsPerBlock * blocksPerGrid)) {
					// Free curand generators and reallocate with a larger size
					// Allocate the nearest power of two generators
					cudaSafeCall(cudaFree(curandGenerators));

					numGenerators = 1;
					while (numGenerators < threadsPerBlock * blocksPerGrid) numGenerators <<= 1;

	#ifdef LIBRAPID_CUDA_STREAM
					cudaSafeCall(cudaMallocAsync(&curandGenerators, sizeof(curandState_t) * numGenerators, cudaStream));
	#else
					cudaSafeCall(cudaMallocAsync(&curandGenerators, sizeof(curandState_t) * numGenerators));
	#endif // LIBRAPID_CUDA_STREAM

					std::string randomKernel = std::string("randomKernel\n") +
											   "#include <stdint.h>\n" +
											   "#include <type_traits>\n" +
											   "#include <" + CUDA_INCLUDE_DIRS + "/curand_kernel.h>\n" +
											   "#include <" + CUDA_INCLUDE_DIRS + "/curand.h>\n\n";

					randomKernel += R"V0G0N(
						__global__
						void populateGenerators(curandState_t *states, int64_t numStates) {
							int64_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
							curand_init()V0G0N";
					randomKernel += std::to_string(getSeed(op)) + ", kernelIndex, 0, &states[kernelIndex]);}\n";

					static jitify::JitCache randomCache;
					jitify::Program randomProgram = kernelCache.program(randomKernel, cudaHeaders, cudaParams);

	#ifdef LIBRAPID_CUDA_STREAM
					jitifyCall(randomProgram.kernel("populateGenerators")
									   .instantiate()
									   .configure(grid, block, 0, cudaStream)
									   .launch(curandGenerators, numGenerators));
	#else
					jitifyCall(program.kernel("populateGenerators")
						.instantiate()
						.configure(grid, block)
						.launch(curandGenerators, numGenerators));
	#endif // LIBRAPID_CUDA_STREAM
				}

				std::visit([&](auto *a, auto *b) {
					using A = typename std::remove_pointer<decltype(a)>::type;
					using B = typename std::remove_pointer<decltype(b)>::type;

					if (op.name == "fillRandom" || op.name == "fillRandomComplex") {
	#ifdef LIBRAPID_CUDA_STREAM
						jitifyCall(program.kernel("unaryFuncTrivial")
										   .instantiate(Type<A>(), Type<B>())
										   .configure(grid, block, 0, cudaStream)
										   .launch(a, b, elems, curandGenerators));
	#else
						jitifyCall(program.kernel("unaryFuncTrivial")
						.instantiate(Type<A>(), Type<B>())
						.configure(grid, block)
						.launch(a, b, elems, curandGenerators));
	#endif // LIBRAPID_CUDA_STREAM
					} else {
#ifdef LIBRAPID_CUDA_STREAM
						jitifyCall(program.kernel("unaryFuncTrivial")
										   .instantiate(Type<A>(), Type<B>())
										   .configure(grid, block, 0, cudaStream)
										   .launch(a, b, elems, nullptr));
#else
						jitifyCall(program.kernel("unaryFuncTrivial")
							.instantiate(Type<A>(), Type<B>())
							.configure(grid, block)
							.launch(a, b, elems, nullptr));
#endif // LIBRAPID_CUDA_STREAM
					}
				}, dst.data, src.data);
			}
#else
			else {
				throw std::runtime_error("CUDA support was not enabled. Invalid operation");
			}
#endif // LIBRAPID_HAS_CUDA
		}
	}

	template<typename FUNC>
	inline void multiarrayUnaryOpComplex(RawArray dst, RawArray src,
										 int64_t elems, const Extent &extent,
										 const Stride &dstStride,
										 const Stride &srcStride,
										 const FUNC &op, bool trivialDst = false) {
		if (dst.location != src.location) {
			// Copy A to be on the same accelerator as B
			RawArray tempSrc = {(int64_t *) nullptr, dst.dtype, dst.location};
			rawArrayMalloc(tempSrc, elems);
			int freeMode = makeSameAccelerator(tempSrc, src, elems);
			multiarrayUnaryOpTrivial(dst, tempSrc, elems, op);
			freeWithMode(tempSrc, freeMode);
		} else {
			// Locations are equal
			if (dst.location == Accelerator::CPU) {
				// Iterate over the array using its stride and extent

				// Counters
				int64_t idim = 0;
				int64_t ndim = extent.ndim();

				// Create pointers here so repeated function calls
				// are not needed
				static int64_t rawExtent[LIBRAPID_MAX_DIMS];
				static int64_t rawDstStride[LIBRAPID_MAX_DIMS];
				static int64_t rawSrcStride[LIBRAPID_MAX_DIMS];

				for (int64_t i = 0; i < ndim; ++i) {
					rawExtent[ndim - i - 1] = extent.raw()[i];
					rawDstStride[ndim - i - 1] = dstStride[i];
					rawSrcStride[ndim - i - 1] = srcStride[i];
				}

				std::visit([&](auto *dstData, auto *srcData) {
					using A = std::remove_pointer<decltype(dstData)>;
					using B = std::remove_pointer<decltype(srcData)>;
					int64_t dstIndex = 0, srcIndex = 0;
					int64_t coord[LIBRAPID_MAX_DIMS]{};

					do {
						// *dstData = op(*srcData);
						dstData[dstIndex] = op(srcData[srcIndex], srcIndex);

						for (idim = 0; idim < ndim; ++idim) {
							if (++coord[idim] == rawExtent[idim]) {
								coord[idim] = 0;
								// srcData = srcData - (rawExtent[idim] - 1) * rawSrcStride[idim];
								// dstData = dstData - (rawExtent[idim] - 1) * rawDstStride[idim];
								srcIndex = srcIndex - (rawExtent[idim] - 1) * rawSrcStride[idim];
								dstIndex = dstIndex - (rawExtent[idim] - 1) * rawDstStride[idim];
							} else {
								// srcData = srcData + rawSrcStride[idim];
								// dstData = dstData + rawDstStride[idim];
								srcIndex = srcIndex + rawSrcStride[idim];
								dstIndex = dstIndex + rawDstStride[idim];
								break;
							}
						}
					} while (idim < ndim);
				}, dst.data, src.data);
			}
#ifdef LIBRAPID_HAS_CUDA
			else {
				using jitify::reflection::Type;

				// Buffers for extent and strides
				int64_t dims = extent.ndim();
				int64_t *deviceExtent, *deviceDstStride = nullptr, *deviceSrcStride;

				// Copy all the required data, including extents and strides
#ifdef LIBRAPID_CUDA_STREAM
				cudaSafeCall(cudaMallocAsync(&deviceExtent, sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaStream));
				if (!trivialDst)
					cudaSafeCall(cudaMallocAsync(&deviceDstStride, sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaStream));
				cudaSafeCall(cudaMallocAsync(&deviceSrcStride, sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaStream));

				cudaSafeCall(cudaMemcpyAsync(deviceExtent, extent.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS,
											 cudaMemcpyHostToDevice, cudaStream));
				if (!trivialDst)
					cudaSafeCall(cudaMemcpyAsync(deviceDstStride, dstStride.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS,
												 cudaMemcpyHostToDevice, cudaStream));
				cudaSafeCall(cudaMemcpyAsync(deviceSrcStride, srcStride.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS,
											 cudaMemcpyHostToDevice, cudaStream));
#else
				cudaSafeCall(cudaMalloc(&deviceExtent, sizeof(int64_t) * LIBRAPID_MAX_DIMS));
				if (!trivialDst) cudaSafeCall(cudaMalloc(&deviceDstStride, sizeof(int64_t) * LIBRAPID_MAX_DIMS));
				cudaSafeCall(cudaMalloc(&deviceSrcStride, sizeof(int64_t) * LIBRAPID_MAX_DIMS));

				cudaSafeCall(cudaMemcpy(deviceExtent, extent.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaMemcpyHostToDevice));
				if (!trivialDst) cudaSafeCall(cudaMemcpy(deviceDstStride, dstStride.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaMemcpyHostToDevice));
				cudaSafeCall(cudaMemcpy(deviceSrcStride, srcStride.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaMemcpyHostToDevice));
#endif // LIBRAPID_CUDA_STREAM

				std::string kernel = "unaryKernelComplex\n";
				kernel += "#include <stdint.h>\n"
						  "#include <type_traits>\n"
						  "#include <" CUDA_INCLUDE_DIRS "/curand_kernel.h>\n"
						  "#include <" CUDA_INCLUDE_DIRS "/curand.h>\n\n";

				kernel += "const int64_t LIBRAPID_MAX_DIMS = "
						  + std::to_string(LIBRAPID_MAX_DIMS) + ";";
				kernel += R"V0G0N(
					__device__
					inline int64_t indexToIndex(uint64_t index,
											   const int64_t *__restrict shape,
											   const int64_t *__restrict strides,
											   uint64_t dims)
					{
						int64_t products[LIBRAPID_MAX_DIMS];
						int64_t prod = 1;
						for (int64_t i = dims - 1; i >= 0; --i)
						{
							products[i] = prod;
							prod *= shape[i];
						}

						int64_t tmp = index;
						int64_t res = 0;
						int64_t tmp2;

						for (int64_t i = 0; i < dims; ++i)
						{
							tmp2 = tmp / products[i];
							res = res + strides[i] * tmp2;
							tmp = tmp - tmp2 * products[i];
						}

						return res;
					}
					)V0G0N";

				kernel += complexHpp;

				kernel += R"V0G0N(
					template<typename A>
					__device__
					inline auto )V0G0N";

				kernel += op.name;

				kernel += R"V0G0N((A &a, int64_t indexA, curandState_t *_curandState)
					{
					)V0G0N";

				kernel += op.kernel;

				kernel += R"V0G0N(}

					template<typename T_DST, typename T_SRC>
					__global__
						void unaryFuncComplex(T_DST * __restrict dstData,
											  const T_SRC * __restrict srcData, int64_t size,
											  const int64_t extent[LIBRAPID_MAX_DIMS],
											  const int64_t dstStride[LIBRAPID_MAX_DIMS],
											  const int64_t srcStride[LIBRAPID_MAX_DIMS],
											  int64_t dims,
											  curandState_t *_curandStates = nullptr)
					{
						uint64_t kernelIndex = blockDim.x * blockIdx.x
											   + threadIdx.x;

						uint64_t srcIndex = indexToIndex(kernelIndex, extent, srcStride, dims);
					)V0G0N";

				if (trivialDst)
					kernel += "uint64_t dstIndex = kernelIndex;";
				else
					kernel += "uint64_t dstIndex = indexToIndex(kernelIndex, extent, dstStride, dims);";

				kernel += R"V0G0N(
						if (kernelIndex < size) {
					)V0G0N";

				kernel += "dstData[dstIndex] = " + op.name +
						  "(srcData[srcIndex], srcIndex, _curandStates + kernelIndex);";
				kernel += "\n}\n}";

				static jitify::JitCache kernelCache;
				jitify::Program program = kernelCache.program(kernel, cudaHeaders, cudaParams);

				unsigned int threadsPerBlock, blocksPerGrid;

				// Use 1 to 512 threads per block
				if (elems < 512) {
					threadsPerBlock = (unsigned int) elems;
					blocksPerGrid = 1;
				} else {
					threadsPerBlock = 512;
					blocksPerGrid = ceil(double(elems) / double(threadsPerBlock));
				}

				dim3 grid(blocksPerGrid);
				dim3 block(threadsPerBlock);

				static curandState_t *curandGenerators = nullptr;
				static int64_t numGenerators = 0;
				if ((op.name == "fillRandom" || op.name == "fillRandomComplex") &&
					(numGenerators < threadsPerBlock * blocksPerGrid)) {
					// Free curand generators and reallocate with a larger size
					// Allocate the nearest power of two generators
					cudaSafeCall(cudaFree(curandGenerators));

					numGenerators = 1;
					while (numGenerators < threadsPerBlock * blocksPerGrid) numGenerators <<= 1;

	#ifdef LIBRAPID_CUDA_STREAM
					cudaSafeCall(cudaMallocAsync(&curandGenerators, sizeof(curandState_t) * numGenerators, cudaStream));
	#else
					cudaSafeCall(cudaMallocAsync(&curandGenerators, sizeof(curandState_t) * numGenerators));
	#endif // LIBRAPID_CUDA_STREAM

					std::string randomKernel = std::string("randomKernel\n") +
											   "#include <stdint.h>\n" +
											   "#include <type_traits>\n" +
											   "#include <" + CUDA_INCLUDE_DIRS + "/curand_kernel.h>\n" +
											   "#include <" + CUDA_INCLUDE_DIRS + "/curand.h>\n\n";

					randomKernel += R"V0G0N(
						__global__
						void populateGenerators(curandState_t *states, int64_t numStates) {
							int64_t kernelIndex = blockDim.x * blockIdx.x + threadIdx.x;
							curand_init()V0G0N";
					randomKernel += std::to_string(getSeed(op)) + ", kernelIndex, 0, &states[kernelIndex]);}\n";

					static jitify::JitCache randomCache;
					jitify::Program randomProgram = kernelCache.program(randomKernel, cudaHeaders, cudaParams);

	#ifdef LIBRAPID_CUDA_STREAM
					jitifyCall(randomProgram.kernel("populateGenerators")
									   .instantiate()
									   .configure(grid, block, 0, cudaStream)
									   .launch(curandGenerators, numGenerators));
	#else
					jitifyCall(program.kernel("populateGenerators")
						.instantiate()
						.configure(grid, block)
						.launch(curandGenerators, numGenerators));
	#endif // LIBRAPID_CUDA_STREAM
				}

				std::visit([&](auto *dstData, auto *srcData) {
					using T_DST = typename std::remove_pointer<decltype(dstData)>::type;
					using T_SRC = typename std::remove_pointer<decltype(srcData)>::type;

#ifdef LIBRAPID_CUDA_STREAM
					jitifyCall(program.kernel("unaryFuncComplex")
									   .instantiate(Type<T_DST>(), Type<T_SRC>())
									   .configure(grid, block, 0, cudaStream)
									   .launch(dstData, srcData, elems, deviceExtent,
											   deviceDstStride, deviceSrcStride,
											   dims, curandGenerators));
#else
					jitifyCall(program.kernel("unaryFuncComplex")
						.instantiate(Type<T_DST>(), Type<T_SRC>())
						.configure(grid, block)
						.launch(dstData, srcData, elems, deviceExtent,
							deviceDstStride, deviceSrcStride,
							dims, curandGenerators));
#endif // LIBRAPID_CUDA_STREAM
				}, dst.data, src.data);

#ifdef LIBRAPID_CUDA_STREAM
				cudaSafeCall(cudaFreeAsync(deviceExtent, cudaStream));
				if (!trivialDst) cudaSafeCall(cudaFreeAsync(deviceDstStride, cudaStream));
				cudaSafeCall(cudaFreeAsync(deviceSrcStride, cudaStream));
#else
				cudaSafeCall(cudaDeviceSynchronize());
				cudaSafeCall(cudaFree(deviceExtent));
				if (!trivialDst) cudaSafeCall(cudaFree(deviceDstStride));
				cudaSafeCall(cudaFree(deviceSrcStride));
#endif // LIBRAPID_CUDA_STREAM
			}
#else
			else {
				throw std::runtime_error("CUDA support was not enabled");
			}
#endif // LIBRAPID_HAS_CUDA
		}
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
	template<typename FUNC>
	inline void multiarrayBinaryOpTrivial(RawArray &dst, const RawArray &srcA,
										  const RawArray &srcB, bool srcAIsScalar,
										  bool srcBIsScalar, int64_t elems,
										  const FUNC &op) {
		if (dst.location != srcA.location || dst.location != srcB.location) {
			// Locations are different, so make A and B have the same
			// accelerator as the result array (C)

			// Copy A and B to be on the same accelerator as the destination
			RawArray tempSrcA = {(int64_t *) nullptr, srcA.dtype, dst.location};
			RawArray tempSrcB = {(int64_t *) nullptr, srcB.dtype, dst.location};

			// Allocate memory for temporary sources
			// rawArrayMalloc(tempSrcA, srcAIsScalar ? 1 : elems);
			// rawArrayMalloc(tempSrcB, srcBIsScalar ? 1 : elems);

			// Copy the data from the original sources to the adjusted sources
			int freeSrcA = makeSameAccelerator(tempSrcA, srcA, srcAIsScalar ? 1 : elems);
			int freeSrcB = makeSameAccelerator(tempSrcB, srcB, srcBIsScalar ? 1 : elems);

			// Apply the operation again, using the new sources
			multiarrayBinaryOpTrivial(dst, tempSrcA, tempSrcB, srcAIsScalar, srcBIsScalar, elems, op);

			// Free the allocated data -- may not be freed if a pointer was
			// simply copied. See the different free modes further up in this
			// file
			freeWithMode(tempSrcA, freeSrcA);
			freeWithMode(tempSrcB, freeSrcB);
		} else {
			// Locations are the same, so apply a single, unified operation
			if (dst.location == Accelerator::CPU) {
				std::visit([&](auto *dstData, auto *srcDataA, auto *srcDataB) {
					auto tempOp = op;
					auto tempElems = elems;

					// Typenames which are useful for casting and checking
					using C = typename std::remove_pointer<decltype(dstData)>::type;
					using A = typename std::remove_pointer<decltype(srcDataA)>::type;
					using B = typename std::remove_pointer<decltype(srcDataB)>::type;

					if (srcAIsScalar) {
						// Use *a rather than a[i]
						if (elems < 2500) {
							for (int64_t i = 0; i < elems; ++i)
								dstData[i] = static_cast<C>(tempOp(*srcDataA, srcDataB[i], 0, i));
						} else {
#pragma omp parallel for shared(dstData, srcDataA, srcDataB, tempElems, tempOp) default(none)
							for (int64_t i = 0; i < tempElems; ++i) {
								dstData[i] = static_cast<C>(tempOp(*srcDataA, srcDataB[i], 0, i));
							}
						}
					} else if (srcBIsScalar) {
						// Use *b rather than b[i]
						if (elems < 2500) {
							for (int64_t i = 0; i < tempElems; ++i)
								dstData[i] = static_cast<C>(tempOp(srcDataA[i], *srcDataB, i, 0));
						} else {
#pragma omp parallel for shared(dstData, srcDataA, srcDataB, tempElems, tempOp) default(none)
							for (int64_t i = 0; i < tempElems; ++i) {
								dstData[i] = static_cast<C>(tempOp(srcDataA[i], *srcDataB, i, 0));
							}
						}
					} else {
						// Use a[i] and b[i]
						if constexpr (std::is_same_v<A, double> && std::is_same_v<B, double> &&
									  std::is_same_v<C, double>) {
							vcl::Vec8d a, b;
							int64_t i = 0;
							auto tmpSrcA = (double *__restrict) srcDataA;
							auto tmpSrcB = (double *__restrict) srcDataB;
							auto tmpDst = (double *__restrict) dstData;

							if (elems < 2500) {
								for (i = 0; i < tempElems - 7; i += 8) {
									a.load(tmpSrcA + i);
									b.load(tmpSrcB + i);
									vcl::Vec8d c = tempOp(a, b, i, i);
									c.store(tmpDst + i);
								}
							} else {
#pragma omp parallel for shared(tmpDst, tmpSrcA, tmpSrcB, tempElems, tempOp, i) private(a, b) default(none)
								for (i = 0; i < tempElems - 7; i += 8) {
									a.load(tmpSrcA + i);
									b.load(tmpSrcB + i);
									vcl::Vec8d c = tempOp(a, b, i, i);
									c.store(tmpDst + i);
								}
							}

							int64_t diff = tempElems - i;
							if (diff > 0) {
								a.load_partial((int) diff, tmpSrcA + i);
								b.load_partial((int) diff, tmpSrcB + i);
								vcl::Vec8d c = tempOp(a, b, i, i);
								c.store_partial((int) diff, tmpDst + i);
							}
						} else if constexpr (std::is_same_v<A, float> && std::is_same_v<B, float> &&
											 std::is_same_v<C, float>) {
							vcl::Vec16f a, b;
							int64_t i = 0;
							auto tmpSrcA = (float *__restrict) srcDataA;
							auto tmpSrcB = (float *__restrict) srcDataB;
							auto tmpDst = (float *__restrict) dstData;

							if (elems < 2500) {
								for (i = 0; i < tempElems - 15; i += 16) {
									a.load(tmpSrcA + i);
									b.load(tmpSrcB + i);
									vcl::Vec16f c = tempOp(a, b, i, i);
									c.store(tmpDst + i);
								}
							} else {
#pragma omp parallel for shared(tmpDst, tmpSrcA, tmpSrcB, tempElems, tempOp, i) private(a, b) default(none)
								for (i = 0; i < tempElems - 15; i += 16) {
									a.load(tmpSrcA + i);
									b.load(tmpSrcB + i);
									vcl::Vec16f c = tempOp(a, b, i, i);
									c.store(tmpDst + i);
								}
							}

							int64_t diff = tempElems - i;
							if (diff > 0) {
								a.load_partial((int) diff, tmpSrcA + i);
								b.load_partial((int) diff, tmpSrcB + i);
								vcl::Vec16f c = tempOp(a, b, i, i);
								c.store_partial((int) diff, tmpDst + i);
							}
						} else if (elems < 2500) {
							for (int64_t i = 0; i < tempElems; ++i) {
								dstData[i] = static_cast<C>(tempOp(srcDataA[i], srcDataB[i], i, i));
							}
						} else {
#pragma omp parallel for shared(dstData, srcDataA, srcDataB, tempElems, tempOp) default(none)
							for (int64_t i = 0; i < tempElems; ++i) {
								dstData[i] = static_cast<C>(tempOp(srcDataA[i], srcDataB[i], i, i));
							}
						}
					}
				}, dst.data, srcA.data, srcB.data);
			}
#ifdef LIBRAPID_HAS_CUDA
			else {
				using jitify::reflection::Type;

				std::string kernel = "binaryKernelTrivial\n";
				kernel += "__constant__ int LIBRAPID_MAX_DIMS = "
						  + std::to_string(LIBRAPID_MAX_DIMS) + ";\n";
				kernel += "#include <stdint.h>\n"
						  "#include <type_traits>\n"
						  "#include <" CUDA_INCLUDE_DIRS "/curand_kernel.h>\n"
						  "#include <" CUDA_INCLUDE_DIRS "/curand.h>\n\n";

				kernel += complexHpp;

				// Insert the user-defined kernel into the main GPU kernel
				kernel += R"V0G0N(
					template<typename A, typename B>
					__device__
					inline auto )V0G0N";

				kernel += op.name;

				kernel += R"V0G0N((A &a, B &b, int64_t indexA, int64_t indexB)
					{
					)V0G0N";

				kernel += op.kernel;

				kernel += R"V0G0N(}

                    template<typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
                    __global__
                    inline T random(T lower, T upper, uint64_t seed = -1) {
                        // Random floating point value in range [lower, upper)

                        static std::uniform_real_distribution<T> distribution(0., 1.);
                        static std::mt19937 generator(seed == (uint64_t) -1 ? (unsigned int) (seconds() * 10) : seed);
                        return lower + (upper - lower) * distribution(generator);
                    }

                    template<typename T, typename std::enable_if<!std::is_floating_point<T>::value, int>::type = 0>
                    __global__
                    inline T random(T lower, T upper, uint64_t seed = -1) {
                        // Random integral value in range [lower, upper]
                        return (T) random((double) (lower - (lower < 0 ? 1 : 0)), (double) upper + 1, seed);
                    }

					template<typename T_DST, typename T_SRCA, typename T_SRCB>
					__global__
					void binaryFuncTrivial(T_DST *__restrict dstData,
										   const T_SRCA *__restrict srcA,
										   const T_SRCB *__restrict srcB,
										   int64_t size)
					{
						const int64_t kernelIndex = blockDim.x * blockIdx.x
												   + threadIdx.x;

						if (kernelIndex < size) {
					)V0G0N";

				if (srcAIsScalar)
					kernel += "dstData[kernelIndex] = " + op.name + "(*srcA, srcB[kernelIndex], 0, kernelIndex);";
				else if (srcBIsScalar)
					kernel += "dstData[kernelIndex] = " + op.name + "(srcA[kernelIndex], *srcB, kernelIndex, 0);";
				else
					kernel += "dstData[kernelIndex] = " + op.name +
							  "(srcA[kernelIndex], srcB[kernelIndex], kernelIndex, kernelIndex);";
				kernel += "\n}\n}";

				static const std::vector<std::string> params = {
						"--disable-warnings", "-std=c++17", std::string("-I \"") + CUDA_INCLUDE_DIRS + "\""
				};

				static jitify::JitCache kernelCache;
				jitify::Program program = kernelCache.program(kernel, cudaHeaders, params);

				int64_t threadsPerBlock, blocksPerGrid;

				// Use 1 to 512 threads per block
				if (elems < 512) {
					threadsPerBlock = elems;
					blocksPerGrid = 1;
				} else {
					threadsPerBlock = 512;
					blocksPerGrid = ceil(double(elems) / double(threadsPerBlock));
				}

				dim3 grid(blocksPerGrid);
				dim3 block(threadsPerBlock);

				std::visit([&](auto *__restrict dstData,
							   auto *__restrict srcDataA,
							   auto *__restrict srcDataB) {
					using T_DST = typename std::remove_pointer_t<decltype(dstData)>;
					using T_SRCA = typename std::remove_pointer_t<decltype(srcDataA)>;
					using T_SRCB = typename std::remove_pointer_t<decltype(srcDataB)>;

					// auto thing = (T_DST) 0;
					// std::cout << "THING: " << thing << "\n";

#ifdef LIBRAPID_CUDA_STREAM
					jitifyCall(program.kernel("binaryFuncTrivial")
									   .instantiate(Type<T_DST>(), Type<T_SRCA>(), Type<T_SRCB>())
									   .configure(grid, block, 0, cudaStream)
									   .launch(dstData, srcDataA, srcDataB, elems));
#else
					jitifyCall(program.kernel("binaryFuncTrivial")
						.instantiate(Type<T_DST>(), Type<T_SRCA>(), Type<T_SRCB>())
						.configure(grid, block)
						.launch(dstData, srcDataA, srcDataB, elems));
#endif // LIBRAPID_CUDA_STREAM
				}, dst.data, srcA.data, srcB.data);
			}
#endif // LIBRAPID_HAS_CUDA
		}
	}

	template<typename FUNC>
	inline void multiarrayBinaryOpComplex(RawArray &dst,
										  const RawArray &srcA,
										  const RawArray &srcB,
										  bool srcAIsScalar,
										  bool srcBIsScalar,
										  const int64_t elems,
										  const Extent &extent,
										  const Stride &strideDst,
										  const Stride &strideSrcA,
										  const Stride &strideSrcB,
										  const FUNC &op) {
		if (dst.location != srcA.location || dst.location != srcB.location) {
			// Locations are different, so make A and B have the same
			// accelerator as the result array (C)

			// Copy A to be on the same accelerator as B
			RawArray tempSrcA = {(int64_t *) nullptr, dst.dtype, dst.location};
			RawArray tempSrcB = {(int64_t *) nullptr, dst.dtype, dst.location};

			// Allocate memory for temporary sources
			rawArrayMalloc(tempSrcA, elems);
			rawArrayMalloc(tempSrcB, elems);

			// Copy the data from the original sources to the adjusted sources
			int freeSrcA = makeSameAccelerator(tempSrcA, srcA, srcAIsScalar ? 1 : elems);
			int freeSrcB = makeSameAccelerator(tempSrcB, srcB, srcBIsScalar ? 1 : elems);

			// Apply the operation again, using the new sources
			multiarrayBinaryOpComplex(dst, tempSrcA, tempSrcB, srcAIsScalar,
									  srcBIsScalar, elems, extent,
									  strideDst, strideSrcA, strideSrcB, op);

			// Free the allocated data -- may not be freed if a pointer was
			// simply copied. See the different free modes further up in this
			// file
			freeWithMode(tempSrcA, freeSrcA);
			freeWithMode(tempSrcB, freeSrcB);
		} else {
			if (dst.location == Accelerator::CPU) {
				// Iterate over the array using its stride and extent
				int64_t coord[LIBRAPID_MAX_DIMS]{};

				// Counters
				int64_t idim = 0;
				int64_t ndim = extent.ndim();

				// Create pointers here so repeated function calls
				// are not needed
				static int64_t rawExtent[LIBRAPID_MAX_DIMS];
				static int64_t _strideDst[LIBRAPID_MAX_DIMS];
				static int64_t _strideSrcA[LIBRAPID_MAX_DIMS];
				static int64_t _strideSrcB[LIBRAPID_MAX_DIMS];

				for (int64_t i = 0; i < ndim; ++i) {
					rawExtent[ndim - i - 1] = extent[i];
					_strideDst[ndim - i - 1] = strideDst[i];
					_strideSrcA[ndim - i - 1] = strideSrcA[i];
					_strideSrcB[ndim - i - 1] = strideSrcB[i];
				}

				std::visit([&](auto *dstData, auto *srcA, auto *srcB) {
					using C = typename std::remove_pointer<decltype(dstData)>::type;
					using A = typename std::remove_pointer<decltype(srcA)>::type;
					using B = typename std::remove_pointer<decltype(srcB)>::type;

					int64_t dstIndex = 0, srcAIndex = 0, srcBIndex = 0;
					if (srcAIsScalar) {
						// Use *a rather than a[i]
						do {
							// *dstData = (C) op(*srcA, *srcB);
							dstData[dstIndex] = (C) op(*srcA, srcB[srcBIndex], 0, srcBIndex);

							for (idim = 0; idim < ndim; ++idim) {
								if (++coord[idim] == rawExtent[idim]) {
									coord[idim] = 0;
									// srcB = srcB - (rawExtent[idim] - 1) * _strideSrcB[idim];
									// dstData = dstData - (rawExtent[idim] - 1) * _strideDst[idim];
									srcBIndex = srcBIndex - (rawExtent[idim] - 1) * _strideSrcB[idim];
									dstIndex = dstIndex - (rawExtent[idim] - 1) * _strideDst[idim];
								} else {
									// srcB = srcB + _strideSrcB[idim];
									// dstData = dstData + _strideDst[idim];
									srcBIndex = srcBIndex + _strideSrcB[idim];
									dstIndex = dstIndex + _strideDst[idim];
									break;
								}
							}
						} while (idim < ndim);
					} else if (srcBIsScalar) {
						// Use *b rather than b[i]
						do {
							// *dstData = (C) op(*srcA, *srcB);
							dstData[dstIndex] = op(srcA[srcAIndex], *srcB, srcAIndex, 0);

							for (idim = 0; idim < ndim; ++idim) {
								if (++coord[idim] == rawExtent[idim]) {
									coord[idim] = 0;
									// srcA = srcA - (rawExtent[idim] - 1) * _strideSrcA[idim];
									// dstData = dstData - (rawExtent[idim] - 1) * _strideDst[idim];
									srcAIndex = srcAIndex - (rawExtent[idim] - 1) * _strideSrcA[idim];
									dstIndex = dstIndex - (rawExtent[idim] - 1) * _strideDst[idim];
								} else {
									// srcA = srcA + _strideSrcA[idim];
									// dstData = dstData + _strideDst[idim];
									srcAIndex = srcAIndex + _strideSrcA[idim];
									dstIndex = dstIndex + _strideDst[idim];
									break;
								}
							}
						} while (idim < ndim);
					} else {
						// Use a[i] and b[i]
						do {
							// *dstData = (C) op(*srcA, *srcB);
							dstData[dstIndex] = (C) op(srcA[srcAIndex], srcB[srcBIndex], srcAIndex, srcBIndex);

							for (idim = 0; idim < ndim; ++idim) {
								if (++coord[idim] == rawExtent[idim]) {
									coord[idim] = 0;
									// dstData = dstData - (rawExtent[idim] - 1) * _strideDst[idim];
									// srcA = srcA - (rawExtent[idim] - 1) * _strideSrcA[idim];
									// srcB = srcB - (rawExtent[idim] - 1) * _strideSrcB[idim];
									dstIndex = dstIndex - (rawExtent[idim] - 1) * _strideDst[idim];
									srcAIndex = srcAIndex - (rawExtent[idim] - 1) * _strideSrcA[idim];
									srcB = srcB - (rawExtent[idim] - 1) * _strideSrcB[idim];
								} else {
									// dstData = dstData + _strideDst[idim];
									// srcA = srcA + _strideSrcA[idim];
									// srcB = srcB + _strideSrcB[idim];
									dstIndex = dstIndex + _strideDst[idim];
									srcAIndex = srcAIndex + _strideSrcA[idim];
									srcBIndex = srcBIndex + _strideSrcB[idim];
									break;
								}
							}
						} while (idim < ndim);
					}
				}, dst.data, srcA.data, srcB.data);
			}
#ifdef LIBRAPID_HAS_CUDA
			else {
				using jitify::reflection::Type;

				// Buffers for extent and strides
				int64_t dims = extent.ndim();
				int64_t *deviceExtent, *deviceStrideSrcA,
						*deviceStrideSrcB, *deviceStrideDst;

#ifdef LIBRAPID_CUDA_STREAM
				cudaSafeCall(cudaMallocAsync(&deviceExtent, sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaStream));
				cudaSafeCall(cudaMallocAsync(&deviceStrideSrcA, sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaStream));
				cudaSafeCall(cudaMallocAsync(&deviceStrideSrcB, sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaStream));
				cudaSafeCall(cudaMallocAsync(&deviceStrideDst, sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaStream));

				cudaSafeCall(cudaMemcpyAsync(deviceExtent, extent.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS,
											 cudaMemcpyHostToDevice, cudaStream));
				cudaSafeCall(cudaMemcpyAsync(deviceStrideSrcA, strideSrcA.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS,
											 cudaMemcpyHostToDevice, cudaStream));
				cudaSafeCall(cudaMemcpyAsync(deviceStrideSrcB, strideSrcB.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS,
											 cudaMemcpyHostToDevice, cudaStream));
				cudaSafeCall(cudaMemcpyAsync(deviceStrideDst, strideDst.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS,
											 cudaMemcpyHostToDevice, cudaStream));
#else
				cudaSafeCall(cudaMalloc(&deviceExtent, sizeof(int64_t) * LIBRAPID_MAX_DIMS));
				cudaSafeCall(cudaMalloc(&deviceStrideSrcA, sizeof(int64_t) * LIBRAPID_MAX_DIMS));
				cudaSafeCall(cudaMalloc(&deviceStrideSrcB, sizeof(int64_t) * LIBRAPID_MAX_DIMS));
				cudaSafeCall(cudaMalloc(&deviceStrideDst, sizeof(int64_t) * LIBRAPID_MAX_DIMS));

				cudaSafeCall(cudaMemcpy(deviceExtent, extent.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaMemcpyHostToDevice));
				cudaSafeCall(cudaMemcpy(deviceStrideSrcA, _strideSrcA.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaMemcpyHostToDevice));
				cudaSafeCall(cudaMemcpy(deviceStrideSrcB, strideSrcB.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaMemcpyHostToDevice));
				cudaSafeCall(cudaMemcpy(deviceStrideDst, strideDst.raw(), sizeof(int64_t) * LIBRAPID_MAX_DIMS, cudaMemcpyHostToDevice));
#endif // LIBRAPID_CUDA_STREAM

				std::string kernel = "binaryKernelComplex\n";
				kernel += "const int64_t LIBRAPID_MAX_DIMS = "
						  + std::to_string(LIBRAPID_MAX_DIMS) + ";\n";

				kernel += "#include <stdint.h>\n"
						  "#include <type_traits>\n"
						  "#include <" CUDA_INCLUDE_DIRS "/curand_kernel.h>\n"
						  "#include <" CUDA_INCLUDE_DIRS "/curand.h>\n\n";

				kernel += R"V0G0N(

                    template<typename T, typename std::enable_if<std::is_floating_point<T>::value, int>::type = 0>
                    __global__
                    inline T random(T lower, T upper, uint64_t seed = -1) {
                        // Random floating point value in range [lower, upper)

                        static std::uniform_real_distribution<T> distribution(0., 1.);
                        static std::mt19937 generator(seed == (uint64_t) -1 ? (unsigned int) (seconds() * 10) : seed);
                        return lower + (upper - lower) * distribution(generator);
                    }

                    template<typename T, typename std::enable_if<!std::is_floating_point<T>::value, int>::type = 0>
                    __global__
                    inline T random(T lower, T upper, uint64_t seed = -1) {
                        // Random integral value in range [lower, upper]
                        return (T) random((double) (lower - (lower < 0 ? 1 : 0)), (double) upper + 1, seed);
                    }

					__device__
					inline int64_t indexToIndex(uint64_t index,
											   const int64_t *__restrict shape,
											   const int64_t *__restrict strides,
											   uint64_t dims)
					{
						int64_t products[LIBRAPID_MAX_DIMS];
						int64_t prod = 1;
						for (int64_t i = dims - 1; i >= 0; --i)
						{
							products[i] = prod;
							prod *= shape[i];
						}

						int64_t tmp = index;
						int64_t res = 0;
						int64_t tmp2;

						for (int64_t i = 0; i < dims; ++i)
						{
							tmp2 = tmp / products[i];
							res = res + strides[i] * tmp2;
							tmp = tmp - tmp2 * products[i];
						}

						return res;
					}

					)V0G0N";

				kernel += complexHpp;

				kernel += R"V0G0N(
					template<typename A, typename B>
					__device__
					inline auto )V0G0N";

				kernel += op.name;

				kernel += R"V0G0N((A &a, B &b, int64_t indexA, int64_t indexB)
					{
					)V0G0N";

				kernel += op.kernel;

				kernel += R"V0G0N(}

					template<typename T_DST, typename T_SRCA, typename T_SRCB>
					__global__
					void binaryFuncComplex(T_DST *__restrict dstData,
										   const T_SRCA *__restrict srcA,
										   const T_SRCB *__restrict srcB,
										   int64_t size,
										   const int64_t extent[LIBRAPID_MAX_DIMS],
										   const int64_t strideC[LIBRAPID_MAX_DIMS],
										   const int64_t strideA[LIBRAPID_MAX_DIMS],
										   const int64_t strideB[LIBRAPID_MAX_DIMS],
										   const int64_t dims)
					{
						int64_t kernelIndex = blockDim.x * blockIdx.x
														+ threadIdx.x;

						int64_t indexDst = indexToIndex(kernelIndex, extent, strideC, dims);
						int64_t indexSrcA = indexToIndex(kernelIndex, extent, strideA, dims);
						int64_t indexSrcB = indexToIndex(kernelIndex, extent, strideB, dims);

						if (kernelIndex < size) {
					)V0G0N";

				if (srcAIsScalar)
					kernel += "dstData[indexDst] = " + op.name + "(*srcA, srcB[indexSrcB], 0, indexSrcB);";
				else if (srcBIsScalar)
					kernel += "dstData[indexDst] = " + op.name + "(srcA[indexSrcA], *srcB, indexSrcA, 0);";
				else
					kernel += "dstData[indexDst] = " + op.name +
							  "(srcA[indexSrcA], srcB[indexSrcB], indexSrcA, indexSrcB);";

				kernel += "\n}\n}";

				static jitify::JitCache kernelCache;
				jitify::Program program = kernelCache.program(kernel, cudaHeaders, cudaParams);

				unsigned int threadsPerBlock, blocksPerGrid;

				// Use 1 to 512 threads per block
				if (elems < 512) {
					threadsPerBlock = (unsigned int) elems;
					blocksPerGrid = 1;
				} else {
					threadsPerBlock = 512;
					blocksPerGrid = ceil(double(elems) / double(threadsPerBlock));
				}

				dim3 grid(blocksPerGrid);
				dim3 block(threadsPerBlock);

				std::visit([&](auto *dstData, auto *srcDataA, auto *srcDataB) {
					using T_SRCA = typename std::remove_pointer<decltype(srcDataA)>::type;
					using T_SRCB = typename std::remove_pointer<decltype(srcDataB)>::type;
					using T_DST = typename std::remove_pointer<decltype(dstData)>::type;

#ifdef LIBRAPID_CUDA_STREAM
					jitifyCall(program.kernel("binaryFuncComplex")
									   .instantiate(Type<T_DST>(), Type<T_SRCA>(), Type<T_SRCB>())
									   .configure(grid, block, 0, cudaStream)
									   .launch(
											   dstData,
											   srcDataA,
											   srcDataB,
											   elems,
											   deviceExtent,
											   deviceStrideDst,
											   deviceStrideSrcA,
											   deviceStrideSrcB,
											   dims));
#else
					jitifyCall(program.kernel("binaryFuncComplex")
						.instantiate(Type<T_DST>(), Type<T_SRCA>(), Type<T_SRCB>())
						.configure(grid, block)
						.launch(dstData, srcDataA, srcDataB, elems,
							deviceExtent,
							deviceStrideSrcA,
							deviceStrideSrcB,
							deviceStrideDst,
							dims));
#endif // LIBRAPID_CUDA_STREAM
				}, dst.data, srcA.data, srcB.data);

#ifdef LIBRAPID_CUDA_STREAM
				cudaSafeCall(cudaFreeAsync(deviceExtent, cudaStream));
				cudaSafeCall(cudaFreeAsync(deviceStrideSrcA, cudaStream));
				cudaSafeCall(cudaFreeAsync(deviceStrideSrcB, cudaStream));
				cudaSafeCall(cudaFreeAsync(deviceStrideDst, cudaStream));
#else
				cudaSafeCall(cudaFree(deviceExtent));
				cudaSafeCall(cudaFree(deviceStrideSrcA));
				cudaSafeCall(cudaFree(deviceStrideSrcB));
				cudaSafeCall(cudaFree(deviceStrideDst));
#endif // LIBRAPID_CUDA_STREAM
			}
#endif // LIBRAPID_HAS_CUDA
		}
	}

}

#endif // LIBRAPID_MUTLIARRAY_OPERATIONS